# NNs-for-Wind-Speed-Prediction
The aim of this project is to perform experimental valuation using a pretrained deep neural network for wind speed prediction. The aforementioned experiments include:
1. Comparisons of NN results with actual data
2. Εxport of vector representations of the input of the neural network from intermediate layers
3. Clustering these vectors using Manhattan metric (L1)

The deep NN which predicts the wind speed has been trained using, as input, results from classical numerical weather forecasting methods and specifically the model Weather Research and Forecasting (WRF). The input of NN constits of predictions for wind intensity, as well as other related features (temperature, humitidy etc), in 6hours time windows, aiming to predict the wind speed of the last hour of the time window. The input is normalized to [0,1].
