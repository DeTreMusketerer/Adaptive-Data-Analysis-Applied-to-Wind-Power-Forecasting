# Adaptive-Data-Analysis-Applied-to-Wind-Power-Forecasting
This GitHub repository contains the scripts for the master's thesis on "Adaptive Data Analysis Applied to Wind Power Forecasting" made by graduate students in Mathematical-Engineering at Aalborg University. The thesis has been made in cooperation with Energinet who also supplied the data used for the numerical experiments.

Authors:	Andreas Anton Andersen, Martin Voigt Vejling, and Morten Stig Kaaber

E-Mails:	{aand17, mvejli17, mkaabe17}@student.aau.dk

In this work time series data is forecasted using autoregressive models, recurrent neural networks, and hybrid decomposition based model combining adaptive data analysis with recurrent neural networks. The forecasting horizon is one hour ahead based on using the wind power production history. The recurrent neural network used for forecasting is a long short-term memory neural network. The adaptive data analysis methods considered in the thesis are
- The EMD [The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis](https://www.researchgate.net/profile/Norden-Huang/publication/215754718_The_empirical_mode_decomposition_and_the_Hilbert_spectrum_for_nonlinear_and_non-stationary_time_series_analysis/links/5a4701af458515f6b0559c64/The-empirical-mode-decomposition-and-the-Hilbert-spectrum-for-nonlinear-and-non-stationary-time-series-analysis.pdf?_sg%5B0%5D=hbscCBnaTclIP3vMw4GvI8V52Lz2j7EuWX8oho3t4wJ5PFsR0TbsSO6cxPZysbBClItCkSlhZ0p60roNDE-CdQ.qIWtegNH2XPQRQPeSmq_bZumcuG5dWbPjsiVKSPYPrmzBFNYOLIk-uaosv3eYX7UPBt0ecVl5dDvkMZdS3MNVQ&_sg%5B1%5D=jhsy6nhniv2PUsMfV_hFW74hU5P5xBth-AH3lYJ4gefI1pGymRx9ZRulnnnNg0WfIIOrXADb5Bzpcgx5RQUYWTG8bFqSXC19ZyyuCfMJx_Jh.qIWtegNH2XPQRQPeSmq_bZumcuG5dWbPjsiVKSPYPrmzBFNYOLIk-uaosv3eYX7UPBt0ecVl5dDvkMZdS3MNVQ&_iepl=)
- The compressive sensing based method of [Data-driven time–frequency analysis](https://pdf.sciencedirectassets.com/272379/1-s2.0-S1063520313X00041/1-s2.0-S1063520312001546/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEBIaCXVzLWVhc3QtMSJIMEYCIQCtq72VuUoBzwpdC%2FqJjHIockHQmZls6VzaPTV2QWdYOwIhALYZ9nybZeGNg1BWalxjGVPNY9dW5QHiuD%2FysR4izcHBKtsECOr%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBBoMMDU5MDAzNTQ2ODY1IgyYH8GtaAUOsxxnTQoqrwTOglriXUvoQWmhtfIyz4UDCo7T9%2Ff4LbyjMibLFOlNO%2BhStNfu3BSN2BcaUM5adZDFD20IGJcH8qmy%2Brb7CmJs%2B%2BXn0gVYx%2B7e6XWreW4s36GRakh0NPmVjDXo67K1IlhsLySixapTNvR%2B7%2F8BgFDMd%2BACwG28viqeLE%2BCvijgp66XrcCFQSK4b7TxHvIhA2WcQGtp1zKXGoDjXc5DFHNv0uN5vsI5C9Nxb8AyY5VD5NRsRxw%2Bt4MijhzhG3yXiV3MaCRtz5%2FI84txxnPLM4Xxyje0diDLqMwoV3oxtVAPFg1JtltjYeGtrew3bxsgWLRjYcJG1RWwAEWMSAeDlFr19kEJKVxhmaFEi%2FnWhMvxQFMjh3XkRf7VlElDEJciuoqREUttmqEc9MIpX8joJbFvBS%2BECL1Lxl3hJQkoNMZEmvaF8pgGnPIVcIdPvhTikeLPhCObdQFP7qXS33pHjnbBlGlnsS0kiF7H84RPQRx8%2F9KHjIVaojyJ78C2vwNTH6YNRNUfjEt13w71UctBgigpfQ1sYcL23yhvM%2BSesnI5%2FmFkn5KGJZUn7KX8LM%2B2TMsraMJj4mMu14qSzZvcBDORSEyKV0idV6ME6xHbqI0nl6u7fpTIdjAr3ijgw4v7fQQh%2BSsDzK3TAeh%2B4RyKDvUnORCBtHx7r2jHJt6mv4nw73Cu9rOVT7OHmWVzsvZ9xti8iNcDtBuY%2BfKEsGWTZdVufyzRLyknXf3y%2FPpVvrinMJWUmJQGOqgBEDPshk1XqCv9NWEbthfeZoDTIcizfhggigGk%2BK8eqRJ9AYQyw6t6Y7dFfLERoUQ7Jg2cSjb9oDYU49ZuUUjO0N4Ijw%2FgchlDxkJBn8ewF7%2BMHeJvcC1AK8y2cn54saGhPzcR0dRVHdfJjxTdqShkrkhN8B2y5CmI9IAB434okZIwT7bn%2FkaRusV0WcMyHvNEXqpedoRARUInGDCQkLG7AC2ICKqJMy4k&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20220519T103808Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYY6VBYBJJ%2F20220519%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=ea864d27a8c880678d38fee0a55981fc4070feea90e3c3e04b66774d9fe2b5b5&hash=434aaa8b1c2d4c9c5631aaff06c9a6cc292ae6f9cb198c91b0bf8ee3cc2bafee&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1063520312001546&tid=spdf-9eb8c7e6-3b2f-4a94-b8fa-fbe50c6ad8d4&sid=25df4783110c624b034abb315c5d9c69d9d4gxrqb&type=client&ua=4d540000575357575f08&rr=70dc33829f8fabce)
- The partial differential equation based method of [A Novel Foward-PDE Approach as an Alternative to Empirical Mode Decomposition](https://arxiv.org/pdf/1802.00835)

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


## Modules


## Scripts


## Usage

