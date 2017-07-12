/*
 *
 * Author: Václav Mach
 *
 * This program is intended to be controlling robotic arm Lynxmotion AL5D through the leap motion sensor
 * This program must be run as user root!
 *
 */

/* ----------------------------------------------------------------------------------------- */
#include <iostream>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <termios.h>    // POSIX terminal control definitions
#include <math.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include "Leap.h"
#include "ros/ros.h"
#include "std_msgs/Float32MultiArray.h"
/* ----------------------------------------------------------------------------------------- */
using namespace Leap;
using namespace std;
int use_leap = 0;
/* ----------------------------------------------------------------------------------------- */
typedef enum {
  X,
  Y,
  Z
} dimension;
/* ----------------------------------------------------------------------------------------- */
typedef enum {
  MIN,
  CENTER,
  MAX,
  RANGE_SIZE    // just to know the size of enum
} range;
/* ----------------------------------------------------------------------------------------- */
typedef enum {
  BASE,         // 0
  SHOULDER,     // 1
  ELBOW,        // 2
  WRIST,        // 3
  WRIST_ROTATE, // 4
  GRIPPER       // 5
} servos;

/* ----------------------------------------------------------------------------------------- */
class Robot_arm {
  public:
    Robot_arm();
    ~Robot_arm();
    void move(int * new_pos);                               // move the arm
    int check_ranges(const int * ranges, int position);
    void set_to_mid();
    int init();
    void send_cmd(const char * cmd);                        // send command
    void set_outfile(int fd);                               // set output file
    static const int num_servos = 6;                        // number of servos
    int * get_pos();                                        // return current position
    static const int ranges[num_servos][RANGE_SIZE];        // min and max ranges of all servos
    unsigned char end[20];                                   // end of each command
    // servo move constants



  private:
    int USB;                                                // USB file descriptor
    static const int bufsize = 256;                         // command buffer size
    int position[num_servos];                               // current position of aech servo
    unsigned char cmd[bufsize];                             // command to send to robotic arm
    int outfile;                                            // output file
};
/* ----------------------------------------------------------------------------------------- */
const int Robot_arm::ranges[Robot_arm::num_servos][RANGE_SIZE] = {  // MIN, CENTER, MAX
  {600,  1500, 2400},                                         // BASE
  {600, 1500, 2200},                                         // SHOULDER
  {600,  1250, 2200},                                         // ELBOW
  {600,  1500, 2400},                                         // WRIST
  {600,  1350, 2400},                                         // WRIST_ROTATE
  {600,  1600, 2400},                                         // GRIPPER
};
/* ----------------------------------------------------------------------------------------- */
void Robot_arm::set_to_mid()
{
  // set all servos to mid position
  unsigned char cmd[] = "#0P1500S200#1P1600S200#2P1400S200#3P1500S200#4P1450S200#5P1400S250\r";     // set to mid slowly
  write(USB, cmd, strlen((char *)cmd));
}
/* ----------------------------------------------------------------------------------------- */
Robot_arm::Robot_arm()
:
  position {     // "center" position for each servo
    1500,
    1500,
    1500,
    1500,
    1350,
    1400
  },
  end {'T', '8', '0', '0', '\r', '\0'},     // 200 ms to complete whole move
  outfile(0)
{
  memset(cmd, 0, bufsize);
}
/* ----------------------------------------------------------------------------------------- */
Robot_arm::~Robot_arm()
{
  close(USB);

  if(outfile)
    close(outfile);
}
/* ----------------------------------------------------------------------------------------- */
int Robot_arm::init()
{
  /* Open File Descriptor */
  USB = open("/dev/ttyUSB0", O_RDWR | O_NONBLOCK | O_NDELAY);

  // Error Handling
  if(USB < 0) {
    cerr << "Error " << errno << " opening " << "/dev/ttyUSB0" << ": " << strerror (errno) << endl;
    return 1;
  }

  // Configure Port
  struct termios tty;
  memset(&tty, 0, sizeof tty);

  // Error Handling
  if(tcgetattr(USB, &tty) != 0) {
    cerr << "Error " << errno << " from tcgetattr: " << strerror(errno) << endl;
    return 1;
  }

  /* Set Baud Rate */
  cfsetospeed (&tty, B9600);
  cfsetispeed (&tty, B9600);

  /* Setting other Port Stuff */
  tty.c_cflag     &=  ~PARENB;            // Make 8n1
  tty.c_cflag     &=  ~CSTOPB;
  tty.c_cflag     &=  ~CSIZE;
  tty.c_cflag     |=  CS8;
  tty.c_cflag     &=  ~CRTSCTS;           // no flow control
  tty.c_lflag     =   0;                  // no signaling chars, no echo, no canonical processing
  tty.c_oflag     =   0;                  // no remapping, no delays
  tty.c_cc[VMIN]  =   0;                  // read doesn't block
  tty.c_cc[VTIME] =   5;                  // 0.5 seconds read timeout

  tty.c_cflag     |=  CREAD | CLOCAL;                  // turn on READ & ignore ctrl lines
  tty.c_iflag     &=  ~(IXON | IXOFF | IXANY);         // turn off s/w flow ctrl
  tty.c_lflag     &=  ~(ICANON | ECHO | ECHOE | ISIG); // make raw
  tty.c_oflag     &=  ~OPOST;                          // make raw

  /* Flush Port, then applies attributes */
  tcflush(USB, TCIFLUSH);

  if(tcsetattr(USB, TCSANOW, &tty) != 0) {
    cerr << "Error " << errno << " from tcsetattr" << endl;
    return 1;
  }

  set_to_mid();

  return 0;
}
/* ----------------------------------------------------------------------------------------- */
int Robot_arm::check_ranges(const int * ranges, int position)
{
  if(position > ranges[MAX] || position < ranges[MIN])
    return 1;     // position of servo i is minimal or maximal

  return 0;     // no error occured
}
/* ----------------------------------------------------------------------------------------- */
void Robot_arm::move(int * new_pos)
{
  int i;
  new_pos[BASE] += 50;
  new_pos[SHOULDER] -= 90;
  new_pos[ELBOW] += 68;

  for(i = 0; i < num_servos; i++) {
    if(position[i] != new_pos[i]) {       // detected move change

      // if(new_pos[i] > ranges[i][MAX]) {
      //   new_pos[i] = ranges[i][MAX];
      //
      // } else if(new_pos[i] < ranges[i][MIN]) {  // moved in safe ranges
      //   new_pos[i] = ranges[i][MIN];
      // }
      if(check_ranges(ranges[i], new_pos[i]) == 0) {  // moved in safe ranges
        position[i] = new_pos[i];         // save the new position
      }
      sprintf((char *)(cmd + (strlen((char *)cmd))), "#%dP%d", i, position[i]);
    }

  }

  if(strlen((char *)cmd) > 0) {               // command is not empty
    strcat((char *)cmd, (char *)end);         // append end of the string to command
    write(USB, cmd, strlen((char *)cmd));     // send command to hand

    if(outfile)     // write to outfile if open
      write(outfile, cmd, strlen((char *)cmd));
  }

  memset(cmd, 0, bufsize);                  // clear buffer
}
/* ----------------------------------------------------------------------------------------- */
int * Robot_arm::get_pos()
{
  return position;
}
/* ----------------------------------------------------------------------------------------- */
void Robot_arm::send_cmd(const char * cmd)
{
  write(USB, cmd, strlen((char *)cmd));
}
/* ----------------------------------------------------------------------------------------- */
void Robot_arm::set_outfile(int fd)
{
  outfile = fd;
}
/* ----------------------------------------------------------------------------------------- */
class CListener : public Listener {
  public:
    virtual void onInit(const Controller&);
    virtual void onConnect(const Controller&);
    virtual void onDisconnect(const Controller&);
    virtual void onExit(const Controller&);
    virtual void onFrame(const Controller&);
    void commandRobot(const std_msgs::Float32MultiArray&);
    void commandRobotUsingIK(float, float, float, float, float, float, float, float);
    virtual void onFocusGained(const Controller&);
    virtual void onFocusLost(const Controller&);
    virtual void onDeviceChange(const Controller&);
    virtual void onServiceConnect(const Controller&);
    virtual void onServiceDisconnect(const Controller&);
    void set_robot_output(int fd);
    void set_filter(double filter);
    void set_ros_publisher(ros::Publisher hp);
    ros::Publisher hand_pub;
    Robot_arm rarm;
    int new_pos[Robot_arm::num_servos];
  private:

    const int timeout = 40000;
    const int filter_const = 100;
    float filter;
    float speed = 2;
    float BASE_HGT = 67.31;      //base hight 2.65"
    float HUMERUS = 146.05;     //shoulder-to-elbow "bone" 5.75"
    float ULNA = 187.325;        //elbow-to-wrist "bone" 7.375"
    float GRIPPER_VAL = 100.00;          //gripper (incl.heavy duty wrist rotate mechanism) length 3.94"
    float hum_sq = HUMERUS*HUMERUS;
    float uln_sq = ULNA*ULNA;

    double last_reward_time = 0;
};
/* ----------------------------------------------------------------------------------------- */
void CListener::onInit(const Controller& controller) {
  int i, * tmp = rarm.get_pos();
  for(i = 0; i < Robot_arm::num_servos; i++)
    new_pos[i] = tmp[i];        // fill new position by current one

  if(rarm.init()) {
    cerr << "Failed to initialize usb connection to robotic hand" << endl;
    exit(1);
  }
  rarm.end[1] = (use_leap == 1 ? '2' : '4');
  cout << "Initialized" << endl;
}
/* ----------------------------------------------------------------------------------------- */
void CListener::onConnect(const Controller& controller) {
  cout << "Leap Motion sensor Connected" << endl;
}
/* ----------------------------------------------------------------------------------------- */
void CListener::onDisconnect(const Controller& controller) {
  cout << "Leap Motion sensor Disconnected" << endl;
}
/* ----------------------------------------------------------------------------------------- */
void CListener::onExit(const Controller& controller) {
  cout << "Exited" << endl;
}
/* ----------------------------------------------------------------------------------------- */
void CListener::onFrame(const Controller& controller) {
  usleep(timeout);        // timeout to move the arm close to real time
  if(use_leap == 0)
    return;
  // Get the most recent frame and report some basic information
  const Frame frame = controller.frame();

  if(frame.hands().count() == 0)    // do nothing if no hand is involved
    return;

  if(frame.hands().count() > 1) {     // do nothing if more than one hand is involved
    //cout << "More than one hand detected, returning to center position" << endl;
    //rarm.set_to_mid();
    return;
  }

  // ------------------------ MAIN LOGIC
  // ------------------------
  // GRIPPER
  float open = 1;
  float reward = 0;
  FingerList fingers = frame.fingers();
  if(!fingers.fingerType(Finger::TYPE_RING)[0].isExtended() && !fingers.fingerType(Finger::TYPE_PINKY)[0].isExtended()
    && ros::Time::now().toSec() - last_reward_time > 2) {
      reward = 1;
      last_reward_time = ros::Time::now().toSec();
  }
  if((!fingers.fingerType(Finger::TYPE_THUMB)[0].isExtended() || !fingers.fingerType(Finger::TYPE_INDEX)[0].isExtended()))
    open = 0;
  // if(frame.fingers().extended().count() <= 4 )
  //   open = 0;

  // float sum_angle_between_fingers = 0;
  // for(int i = 1; i < 4; i++) {
  //   sum_angle_between_fingers += frame.fingers()[i].direction().angleTo(fingers()[i+1].direction());
  // }
  // // cout << fingers.fingerType(Finger::TYPE_THUMB).isExtended() << fingers.fingerType(Finger::TYPE_INDEX).isExtended()<< fingers.fingerType(2).isExtended()<< fingers.fingerType(3).isExtended()<< fingers.fingerType(4).isExtended() << endl;
  // cout << sum_angle_between_fingers << endl;
  //cout <<  "new_pos[GRIPPER]: " << new_pos[GRIPPER] << endl;

  HandList hands = frame.hands();
  for (HandList::const_iterator hl = hands.begin(); hl != hands.end(); ++hl) {
    // Get the first hand
    const Hand hand = *hl;
    Arm arm = hand.arm();
    // const Vector direction = hand.direction();
    // const Vector normal = hand.palmNormal();
    const FingerList fingers = hand.fingers();

    /* z is height, y is distance from base center out, x is side to side. y,z can only be positive */
    float x = speed * hand.palmPosition()[Z] + 0;
    float y = speed * -hand.palmPosition()[X] + 250;
    float z = speed * (hand.palmPosition()[Y] - 300) + 100;
    float roll = hand.palmNormal().roll();
    float pitch = hand.palmNormal().pitch();
    float yaw = hand.palmNormal().yaw();
    float direction_roll = hand.direction().roll();
    float direction_pitch = hand.direction().pitch();
    float direction_yaw = hand.direction().yaw();

    float wrist_degree = roll;
    float wrist_rotate_degree = direction_yaw;

    commandRobotUsingIK(x, y, z, wrist_degree, wrist_rotate_degree, open, reward, 0);
  }

}

void CListener::commandRobotUsingIK(float x, float y, float z, float wrist_degree, float wrist_rotate_degree, float open, float reward, float human) {

      // cout <<  "\tx: " << x <<  "\ty: " << y <<  "\tz: " << z << endl;
      // cout <<  "\troll: " << roll <<  "\tpitch: " << pitch <<  "\tyaw: " << yaw << endl;
      // cout <<  "\troll: " << direction_roll <<  "\tpitch: " << direction_pitch <<  "\tyaw: " << direction_yaw << endl;
      float grip_angle_d = 0;
      float grip_angle_r = grip_angle_d  * M_PI / 180.0;    //grip angle in radians for use in calculations
      /* Base angle and radial distance from x,y coordinates */
      float bas_angle_r = atan2( x, y );
      float rdist = sqrt(( x * x ) + ( y * y ));
      /* rdist is y coordinate for the arm */
      y = rdist;
      /* Grip offsets calculated based on grip angle */
      float grip_off_z = ( sin( grip_angle_r )) * GRIPPER_VAL;
      float grip_off_y = ( cos( grip_angle_r )) * GRIPPER_VAL;
      /* Wrist position */
      float wrist_z = ( z - grip_off_z ) - BASE_HGT;
      float wrist_y = y - grip_off_y;
      /* Shoulder to wrist distance ( AKA sw ) */
      float s_w = ( wrist_z * wrist_z ) + ( wrist_y * wrist_y );
      float s_w_sqrt = sqrt( s_w );
      /* s_w angle to ground */
      float a1 = atan2( wrist_z, wrist_y );
      /* s_w angle to humerus */
      float cos = (( hum_sq - uln_sq ) + s_w ) / ( 2 * HUMERUS * s_w_sqrt );
      float a2 = acos(cos);
      /* shoulder angle */
      float shl_angle_r = a1 + a2;
      // if (isnan(shl_angle_r) || isinf(shl_angle_r))
      //   return;
      float shl_angle_d = shl_angle_r * 180.0 / M_PI;
      /* elbow angle */
      float elb_angle_r = acos(( hum_sq + uln_sq - s_w ) / ( 2 * HUMERUS * ULNA ));
      float elb_angle_d = elb_angle_r * 180.0 / M_PI;
      float elb_angle_dn = -( 180.0 - elb_angle_d );
      /* wrist angle */
      float wri_angle_d = ( grip_angle_d - elb_angle_dn ) - shl_angle_d;

      new_pos[BASE] = Robot_arm::ranges[BASE][CENTER]- int((( bas_angle_r * 180.0 / M_PI) * 11.11 ));
      new_pos[SHOULDER] = Robot_arm::ranges[SHOULDER][CENTER] + int((( shl_angle_d - 90.0 ) * 6.6 ));
      new_pos[ELBOW] = Robot_arm::ranges[ELBOW][CENTER] -  int((( elb_angle_d - 90.0 ) * 6.6 ));
      new_pos[WRIST] = Robot_arm::ranges[WRIST][CENTER] + int(( wri_angle_d  * 11.1 / 2 ) - (wrist_degree) * 1000 - 600);
      new_pos[WRIST_ROTATE] = Robot_arm::ranges[WRIST_ROTATE][CENTER] + int((( bas_angle_r * 180.0 / M_PI) * 11.11 ) + wrist_rotate_degree * 1200);
      new_pos[GRIPPER] = Robot_arm::ranges[GRIPPER][MAX] - int( open * 1400);  // move GRIPPER in safe ranges

      cout << " " << new_pos[BASE] << " " << new_pos[SHOULDER] << " " << new_pos[ELBOW] << " " << new_pos[WRIST_ROTATE] << " " << new_pos[WRIST] <<  " " << new_pos[GRIPPER] << endl;

      // Publish hand information on ROS topic
      if(ros::ok() && !isnan(shl_angle_r) && !isinf(shl_angle_r) && new_pos[0] > 0 && new_pos[1] > 0 && new_pos[2] > 0 && new_pos[3] > 0 && new_pos[4] > 0
          && new_pos[0] < 4000 && new_pos[1] < 4000 && new_pos[2] < 4000 && new_pos[3] < 4000 && new_pos[4] < 4000) {
        vector<float> vec1 = {reward, human, open, new_pos[0]/float(2000), new_pos[1]/float(2000), new_pos[2]/float(2000), new_pos[3]/float(2000), new_pos[4]/float(2000), 0};
        std_msgs::Float32MultiArray msg;

        // set up dimensions
        msg.layout.dim.push_back(std_msgs::MultiArrayDimension());
        msg.layout.dim[0].size = vec1.size();
        msg.layout.dim[0].stride = 1;
        msg.layout.dim[0].label = "demonstration_info"; // or whatever name you typically use to index vec1

        // copy in the data
        msg.data.clear();
        msg.data.insert(msg.data.end(), vec1.begin(), vec1.end());
        hand_pub.publish(msg);
        ros::spinOnce();

        // filtration
        if(filter != 0) {
          int * tmp_pos = rarm.get_pos();
          // cout << " " << new_pos[0] << " " << new_pos[1] << " " << new_pos[2] << " " << new_pos[3] << " " << new_pos[4] <<  " " << new_pos[5] << endl;
          // cout << " " << tmp_pos[0] << " " << tmp_pos[1] << " " << tmp_pos[2] << " " << tmp_pos[3] << " " << tmp_pos[4] <<  " " << tmp_pos[5] << endl;

          for(int i = 0; i < Robot_arm::num_servos; i++) {
            if(fabs((tmp_pos[i] - new_pos[i]) / (double)filter_const) > filter) {      // move only if greater than filter
              rarm.move(new_pos);
              break;
            }
          }
          return;
        }
        rarm.move(new_pos);
      }


}

void CListener::commandRobot(const std_msgs::Float32MultiArray& msg) {
  new_pos[BASE] = int(msg.data[0] * 2000);
  new_pos[SHOULDER] = int(msg.data[1] * 2000);
  new_pos[ELBOW] = int(msg.data[2] * 2000);
  new_pos[WRIST] = int(msg.data[3] * 2000);
  new_pos[WRIST_ROTATE] = int(msg.data[4] * 2000);
  new_pos[GRIPPER] = Robot_arm::ranges[GRIPPER][MAX] - int( msg.data[5] * 1400);
  // float bas_angle_r = (Robot_arm::ranges[BASE][CENTER] - new_pos[BASE] ) / 180.0 * M_PI / 11.11;
  // new_pos[WRIST_ROTATE] = int(Robot_arm::ranges[WRIST_ROTATE][CENTER]) + int(( bas_angle_r * 180.0 / M_PI) * 11.11 );
  rarm.move(new_pos);
}
/* ----------------------------------------------------------------------------------------- */
void CListener::onFocusGained(const Controller& controller) {
  cout << "Focus Gained" << endl;
}
/* ----------------------------------------------------------------------------------------- */
void CListener::onFocusLost(const Controller& controller) {
  cout << "Focus Lost" << endl;
}
/* ----------------------------------------------------------------------------------------- */
void CListener::onDeviceChange(const Controller& controller) {
}
/* ----------------------------------------------------------------------------------------- */
void CListener::onServiceConnect(const Controller& controller) {
}
/* ----------------------------------------------------------------------------------------- */
void CListener::onServiceDisconnect(const Controller& controller) {
}
/* ----------------------------------------------------------------------------------------- */
void CListener::set_robot_output(int fd)
{
  rarm.set_outfile(fd);
}
/* ----------------------------------------------------------------------------------------- */
void CListener::set_filter(double f)
{
  filter = f;
}
/* ----------------------------------------------------------------------------------------- */
void CListener::set_ros_publisher(ros::Publisher hp)
{
  hand_pub = hp;
}
/* ----------------------------------------------------------------------------------------- */
void print_help()
{
  cout << "leap_to_hand [-h] [-l use_leap(0/1)] [-o output_file] [-i input_file] [-f filter_rate ]" << endl;
  cout << "filter rate is decimal number from 0 to 1" << endl;
}
/* ----------------------------------------------------------------------------------------- */
int check_file(int * infile, const char * name)
{
  // check file exists and is readable
  if(access(name, F_OK | R_OK) != 0) {
    cerr << "Error " << errno << " from access: " << strerror(errno) << endl;
    return 1;
  }

  *infile = open(name, O_RDWR);

  if(*infile == -1) {
    cerr << "Error " << errno << " from open: " << strerror(errno) << endl;
    return 1;
  }

  return 0;
}
/* ----------------------------------------------------------------------------------------- */
int create_file(int * outfile, const char * name)
{
  // check file can be created and written
  *outfile = open(name, O_RDWR | O_CREAT | O_TRUNC,  S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);

  if(*outfile == -1) {
    cerr << "Error " << errno << " from open: " << strerror(errno) << endl;
    return 1;
  }

  return 0;
}
/* ----------------------------------------------------------------------------------------- */
void parse_argv(int argc, char ** argv, int* use_leap, int * infile, int * outfile, double * filter)
{
  int c;

  while ((c = getopt (argc, argv, "ho:i:f:l:")) != -1)
    switch (c) {
      case 'h':
        print_help();
        return;
      case 'o':
        if(create_file(outfile, optarg))
          exit(1);
        return;
      case 'i':
        if(check_file(infile, optarg))
          exit(1);
        return;
      case 'l':
        *use_leap = int(strtod(optarg, NULL));
        return;
      case 'f':
        *filter = strtod(optarg, NULL);

        if(*filter < 0 || *filter > 1) {
          cout << "filter value out of range" << endl;
          print_help();
          exit(1);
        }
        break;
      default:
        print_help();
        exit(1);
    }
}
/* ----------------------------------------------------------------------------------------- */
void play(int infile)
{
  Robot_arm rarm;
  size_t bufsize = 256;
  ssize_t ret = 0;
  char * cmd = NULL;
  FILE * in = fdopen(infile, "r");

  if(rarm.init()) {
    cerr << "Failed to initialize usb connection to robotic hand" << endl;
    exit(1);
  }

  rarm.set_to_mid();
  sleep(3);

  ret = getdelim(&cmd, &bufsize, '\r', in);

  while(ret != EOF) {
    rarm.send_cmd(cmd);
    ret = getdelim(&cmd, &bufsize, '\r', in);
  }

  fclose(in);
  close(infile);
}

CListener listener;

void commandCallback(const std_msgs::Float32MultiArray& msg) {
  listener.commandRobot(msg);
}

void moveInfoCallback(const std_msgs::Float32MultiArray& msg) {
  float x = msg.data[8];
  float y = msg.data[9];
  float z = msg.data[10];
  float wrist_degree = msg.data[11];
  float wrist_rotate_degree = msg.data[12];
  float open = msg.data[13];
  float reward = msg.data[14];
  float human = msg.data[15];
  listener.commandRobotUsingIK(x, y, z, wrist_degree, wrist_rotate_degree, open, reward, human);
}

/* ----------------------------------------------------------------------------------------- */
int main(int argc, char ** argv)
{
  int infile = 0, outfile = 0;
  double filter = 0;
  parse_argv(argc, argv, &use_leap, &infile, &outfile, &filter);
  if(infile != 0) {
    play(infile);
    return 0;
  }

  // Create a sample listener and controller

  Controller controller;

  if(outfile != 0)
    listener.set_robot_output(outfile);

  listener.set_filter(filter);

  cout << "Initializing Ros." << endl;
  // Create ROS node
  ros::init(argc, argv, "leap_al5d");
  ros::NodeHandle n;
  controller.addListener(listener);
  ros::Subscriber command_sub;
  ros::Subscriber move_info_sub;
  // Have the sample listener receive events from the controller
  ros::Publisher hand_pub = n.advertise<std_msgs::Float32MultiArray>("leap_al5d_info", 1000);

  listener.set_ros_publisher(hand_pub);
  command_sub = n.subscribe("robot_command", 1000, commandCallback);
  move_info_sub = n.subscribe("move_info", 1000, moveInfoCallback);
  // while(ros::ok()) {
  //   vector<float> vec1 = {0.0, 1.0, 0.750999987, 0.76649999, 0.60449999, 0.46650001, 0.69900, 0.0};
  //   listener.new_pos[BASE] = 0.7509999871253967 * 2000;
  //   listener.new_pos[SHOULDER] = 0.7664999961853027 * 2000;
  //   listener.new_pos[ELBOW] = 0.6044999957084656 * 2000;
  //   listener.new_pos[WRIST] = 0.46650001406669617 * 2000;
  //   listener.new_pos[WRIST_ROTATE] = 0.6990000009536743 * 2000;
  //   listener.new_pos[GRIPPER] = Robot_arm::ranges[GRIPPER][MAX] - int( 1.0 * 1400);  // move GRIPPER in safe ranges
  //   listener.rarm.move(listener.new_pos);
  //   std_msgs::Float32MultiArray msg;
  //
  //   // set up dimensions
  //   msg.layout.dim.push_back(std_msgs::MultiArrayDimension());
  //   msg.layout.dim[0].size = vec1.size();
  //   msg.layout.dim[0].stride = 1;
  //   msg.layout.dim[0].label = "demonstration_info"; // or whatever name you typically use to index vec1
  //
  //   // copy in the data
  //   msg.data.clear();
  //   msg.data.insert(msg.data.end(), vec1.begin(), vec1.end());
  //   listener.hand_pub.publish(msg);
  //   ros::spinOnce();
  // }
  // Keep this process running until Enter is pressed
  // cout << "Press Enter to quit..." << endl;
  // cin.get();
  ros::spin();

  if(use_leap != 1) {
    // Remove the sample listener when done
    controller.removeListener(listener);
  }
  return 0;
}
