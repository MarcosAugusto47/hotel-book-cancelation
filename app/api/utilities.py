import pandas as pd
import numpy as np
import pickle
from keras.models import load_model

with open('models/scaler.pickle', 'rb') as f:
    scaler_fitted = pickle.load(f)

model_NN = load_model('models/model_NN.h5')

def predict_cancellation(new_data):
    # the output is not exactly one or zero, it should be rounded
    predictions = np.round(model_NN.predict(new_data)[0])

    pred_to_label = {0: 'Not Cancelled', 1: 'Cancelled'}

    # Make a list of predictions
    data = []
    for t, pred in zip(new_data, predictions):
        data.append({'pred': int(pred), 'label': pred_to_label[pred]})

    return data

if __name__ == "__main__":
    new_sample = {'lead_time': 142,
                    'arrival_date_week_number': 17,
                    'arrival_date_day_of_month': 22,
                    'stays_in_weekend_nights': 2,
                    'stays_in_week_nights': 3,
                    'adults': 2,
                    'babies': 0,
                    'is_repeated_guest': 0,
                    'previous_cancellations': 0,
                    'previous_bookings_not_canceled': 0,
                    'booking_changes': 0,
                    'days_in_waiting_list': 0,
                    'required_car_parking_spaces': 0,
                    'total_of_special_requests': 0,
                    'month': 1,
                    'day': 18,
                    'hotel_City Hotel': 1,
                    'meal_BB': 1,
                    'meal_HB': 0,
                    'meal_SC': 0,
                    'meal_Undefined': 0,
                    'reserved_room_type_A': 1,
                    'reserved_room_type_B': 0,
                    'reserved_room_type_C': 0,
                    'reserved_room_type_D': 0,
                    'reserved_room_type_E': 0,
                    'reserved_room_type_F': 0,
                    'reserved_room_type_G': 0,
                    'reserved_room_type_H': 0,
                    'distribution_channel_Corporate': 0,
                    'distribution_channel_Direct': 0,
                    'distribution_channel_GDS': 0,
                    'distribution_channel_TA/TO': 1,
                    'customer_type_Contract': 0,
                    'customer_type_Transient': 1,
                    'customer_type_Transient-Party': 0}
    new_sample = pd.DataFrame([new_sample]).values
    new_sample_scaled = scaler_fitted.transform(new_sample)
    predictions = predict_cancellation(new_sample_scaled)                        
    print(predictions)    