# Snowflake generation using genetic algorithms

### Evaluation methods 

For automated evaluation of aesthetics of snowflakes, we're using models from [this repository](https://github.com/idealo/image-quality-assessment)

Code for running GA's: 

```bash
docker build -t nima-cpu . -f Dockerfile.cpu
 ./predict  --docker-image nima-cpu --base-model-name MobileNet --weights-file $(pwd)/models/MobileNet/weights_mobilenet_aesthetic_0.07.hdf5 
```

