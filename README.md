# Adaptive-Data-Analysis-Applied-to-Wind-Power-Forecasting
This GitHub repository contains the scripts for the master's thesis on "Adaptive Data Analysis Applied to Wind Power Forecasting" made by graduate students in Mathematical-Engineering at Aalborg University. The thesis has been made in cooperation with Energinet who also supplied the data used for the numerical experiments.

Authors:	Andreas Anton Andersen, Martin Voigt Vejling, and Morten Stig Kaaber

E-Mails:	andreasantonandersen@gmail.com, martin.vejling@gmail.com, ahkaaber@gmail.com

In this work time series data is forecasted using autoregressive models, recurrent neural networks, and hybrid decomposition based model combining adaptive data analysis with recurrent neural networks. The forecasting horizon is one hour ahead based on using the wind power production history. The recurrent neural network used for forecasting is a long short-term memory neural network. The adaptive data analysis methods considered in the thesis are
- The EMD [The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis](https://www.researchgate.net/profile/Norden-Huang/publication/215754718_The_empirical_mode_decomposition_and_the_Hilbert_spectrum_for_nonlinear_and_non-stationary_time_series_analysis/links/5a4701af458515f6b0559c64/The-empirical-mode-decomposition-and-the-Hilbert-spectrum-for-nonlinear-and-non-stationary-time-series-analysis.pdf?_sg%5B0%5D=hbscCBnaTclIP3vMw4GvI8V52Lz2j7EuWX8oho3t4wJ5PFsR0TbsSO6cxPZysbBClItCkSlhZ0p60roNDE-CdQ.qIWtegNH2XPQRQPeSmq_bZumcuG5dWbPjsiVKSPYPrmzBFNYOLIk-uaosv3eYX7UPBt0ecVl5dDvkMZdS3MNVQ&_sg%5B1%5D=jhsy6nhniv2PUsMfV_hFW74hU5P5xBth-AH3lYJ4gefI1pGymRx9ZRulnnnNg0WfIIOrXADb5Bzpcgx5RQUYWTG8bFqSXC19ZyyuCfMJx_Jh.qIWtegNH2XPQRQPeSmq_bZumcuG5dWbPjsiVKSPYPrmzBFNYOLIk-uaosv3eYX7UPBt0ecVl5dDvkMZdS3MNVQ&_iepl=)
- The compressive sensing based method of [Data-driven time–frequency analysis](https://pdf.sciencedirectassets.com/272379/1-s2.0-S1063520313X00041/1-s2.0-S1063520312001546/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEBIaCXVzLWVhc3QtMSJIMEYCIQCtq72VuUoBzwpdC%2FqJjHIockHQmZls6VzaPTV2QWdYOwIhALYZ9nybZeGNg1BWalxjGVPNY9dW5QHiuD%2FysR4izcHBKtsECOr%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBBoMMDU5MDAzNTQ2ODY1IgyYH8GtaAUOsxxnTQoqrwTOglriXUvoQWmhtfIyz4UDCo7T9%2Ff4LbyjMibLFOlNO%2BhStNfu3BSN2BcaUM5adZDFD20IGJcH8qmy%2Brb7CmJs%2B%2BXn0gVYx%2B7e6XWreW4s36GRakh0NPmVjDXo67K1IlhsLySixapTNvR%2B7%2F8BgFDMd%2BACwG28viqeLE%2BCvijgp66XrcCFQSK4b7TxHvIhA2WcQGtp1zKXGoDjXc5DFHNv0uN5vsI5C9Nxb8AyY5VD5NRsRxw%2Bt4MijhzhG3yXiV3MaCRtz5%2FI84txxnPLM4Xxyje0diDLqMwoV3oxtVAPFg1JtltjYeGtrew3bxsgWLRjYcJG1RWwAEWMSAeDlFr19kEJKVxhmaFEi%2FnWhMvxQFMjh3XkRf7VlElDEJciuoqREUttmqEc9MIpX8joJbFvBS%2BECL1Lxl3hJQkoNMZEmvaF8pgGnPIVcIdPvhTikeLPhCObdQFP7qXS33pHjnbBlGlnsS0kiF7H84RPQRx8%2F9KHjIVaojyJ78C2vwNTH6YNRNUfjEt13w71UctBgigpfQ1sYcL23yhvM%2BSesnI5%2FmFkn5KGJZUn7KX8LM%2B2TMsraMJj4mMu14qSzZvcBDORSEyKV0idV6ME6xHbqI0nl6u7fpTIdjAr3ijgw4v7fQQh%2BSsDzK3TAeh%2B4RyKDvUnORCBtHx7r2jHJt6mv4nw73Cu9rOVT7OHmWVzsvZ9xti8iNcDtBuY%2BfKEsGWTZdVufyzRLyknXf3y%2FPpVvrinMJWUmJQGOqgBEDPshk1XqCv9NWEbthfeZoDTIcizfhggigGk%2BK8eqRJ9AYQyw6t6Y7dFfLERoUQ7Jg2cSjb9oDYU49ZuUUjO0N4Ijw%2FgchlDxkJBn8ewF7%2BMHeJvcC1AK8y2cn54saGhPzcR0dRVHdfJjxTdqShkrkhN8B2y5CmI9IAB434okZIwT7bn%2FkaRusV0WcMyHvNEXqpedoRARUInGDCQkLG7AC2ICKqJMy4k&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20220519T103808Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYY6VBYBJJ%2F20220519%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=ea864d27a8c880678d38fee0a55981fc4070feea90e3c3e04b66774d9fe2b5b5&hash=434aaa8b1c2d4c9c5631aaff06c9a6cc292ae6f9cb198c91b0bf8ee3cc2bafee&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1063520312001546&tid=spdf-9eb8c7e6-3b2f-4a94-b8fa-fbe50c6ad8d4&sid=25df4783110c624b034abb315c5d9c69d9d4gxrqb&type=client&ua=4d540000575357575f08&rr=70dc33829f8fabce)
- The partial differential equation based method of [A Novel Foward-PDE Approach as an Alternative to Empirical Mode Decomposition](https://arxiv.org/pdf/1802.00835)

The thesis is found in "Adaptive Data Analysis Theoretical Results and an Application to Wind Power Forecasting (2022).pdf".

## Dependencies
This project is created with `Python 3.9`

Dependencies:
```
matplotlib 3.4.3
numpy 1.21.2
scipy 1.7.1
pytorch 1.10
EMD-signal 1.2.3
```


## Data
- Data/
    - DK1-1_Capacity.npy
    - test_data.npy
    - training_data.npy
    - test_mesh_q288.npy
    - test_mesh_not_realtime.npy
    - training_mesh_q288.npy
    - train_mesh_not_realtime.npy
    - subtraining_data.npy
    - subtraining_mesh_not_realtime.npy
    - validation_data.npy
    - validation_mesh_not_realtime.npy
    - EMD_full_training_data.npy
    - EMD_full_test_data.npy


## Modules
`HilberHuangTransform.py`
	- Module with functionality used to calculate instantaneous frequency and amplitude using the Hilbert-Huang transform.

`NN_module.py`
	- Module containing functionality used when training and testing neural networks.

`NonlinearMatchingPursuit.py`
	- Module containing the functionality used to make the NMP-EMD and FFT-NMP-EMD.

`PDE_EMD.py`
	- Module containing the functionality used to make the PDE-EMD.

`PerformanceMeasures.py`
	- Module containing functions used to calculate decomposition performance measures.
    
`sVARMAX_Module.py`
	- Module containing the main class for estimation and forecasting with s-ARIMAX and s-VARIMAX models.
    
## Scripts
`LSTM_Validation.py`
	- Script used to train an LSTM model using early stopping.

`LSTM_Test.py`
	- Script used to train and evaluate a LSTM model.

`EMD_decomposition.py`
	- Script used to decompose the wind power data using the EMD method.

`NMP_EMD_decomposition.py`
	- Script used to decompose the wind power data using the FFT-NMP-EMD or NMP-EMD method.

`EMD_LSTM_Validation.py`
	- Script used to train an EMD-LSTM, FFT-NMP-EMD-LSTM, or NMP-EMD-LSTM model using early stopping.

`EMD_LSTM_Test.py`
	- Script used to train and evaluate a EMD-LSTM, FFT-NMP-EMD-LSTM, or NMP-EMD-LSTM model.

`PDE_EMD_decomposition.py`
	- Script used to decompose the wind power data using the PDE-EMD method.

`PDE_EMD_LSTM_Validation.py`
	- Script used to train a PDE-EMD-LSTM model using early stopping.

`PDE_EMD_LSTM_Test.py`
	- Script used to train and evaluate a PDE-EMD-LSTM model.

`Forecast.py`
	- Script used to create a forecast for the oddline EMD-LSTM and the PDE-EMD-LSTM.

`KDE.py`
	- Script used to calculate the kernel density estimate of the errors for the implemented models.

`PDE_EMD_unification.py`
	- The unification procedure for the PDE-EMD.

`PDE_EMD_unified_window_data_analysis.py`
	- Module used to analyse the PDE-EMD before unification.

`PDE_EMD_window_data_analysis.py`
	- Module used to analyse the PDE-EMD after unification.

`Percentile.py`
	- Module containing functionality used to calculate a percentile error for the different forecasting models.

## Usage
To use this GitHub repository follow these steps:

1) Install Python with the dependencies stated in the Dependencies section.
2) Make decompositions using the decomposition scripts.
3) Run early stopping for a model using the validation scripts.
4) Train and test models using the test scripts.

Make sure to set the parameters etc. in the scripts. Note that the neural network models are compatible with GPU computing if CUDA is available and we recommend using a GPU for these computations.
