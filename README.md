# car_classifier_docker
This is a docker run-time to perform a forward pass on my car-classifier ML model shared at this link https://huggingface.co/SriramSridhar78/sriram-car-classifier.


## How to build and run this docker image?

1. docker build -t car-classifier --rm .

2. docker run -e URL=https://barrettjacksoncdn.azureedge.net/staging/carlist/items/Fullsize/Cars/200473/200473_Side_Profile_Web.jpg --rm car-classifier

```
2022-03-02 00:10:06,824 INFO car_classifier <module> Model load time : 1225.6300009903498 msec
2022-03-02 00:10:07,158 INFO car_classifier <module> Infer time : 333.6595419677906 msec
score=0.3637121617794037, index=102.0
Could be Ferrari_California_Convertible_2012 with probability score of 0.3637121617794037.
score=0.24097496271133423, index=100.0
Could be Ferrari_458_Italia_Convertible_2012 with probability score of 0.24097496271133423.
```
