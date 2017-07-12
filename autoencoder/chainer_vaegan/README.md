# chainer-vae-gan

Implementation of [Autoencoding beyond pixels using a learned similarity metric](https://arxiv.org/abs/1512.09300)

## Usage

### Make dataset file

```
$ python src/convert_dataset.py image_dir images.pkl
```

### Train model

```
$ python src/train.py -g 0 -d images.pkl -o model/test --out_image_dir image/out
```

## License

MIT License
