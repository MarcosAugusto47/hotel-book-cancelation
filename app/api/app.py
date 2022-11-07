import json
import pandas as pd
from flask import Flask, jsonify, request
from utilities import scaler_fitted, predict_cancellation

app = Flask(__name__)

covariables = ['lead_time', 'arrival_date_week_number', 'arrival_date_day_of_month',
               'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'babies',
               'is_repeated_guest', 'previous_cancellations',
               'previous_bookings_not_canceled', 'booking_changes',
               'days_in_waiting_list', 'required_car_parking_spaces',
               'total_of_special_requests', 'month', 'day', 'hotel_City Hotel',
               'meal_BB', 'meal_HB', 'meal_SC', 'meal_Undefined',
               'reserved_room_type_A', 'reserved_room_type_B', 'reserved_room_type_C',
               'reserved_room_type_D', 'reserved_room_type_E', 'reserved_room_type_F',
               'reserved_room_type_G', 'reserved_room_type_H',
               'distribution_channel_Corporate', 'distribution_channel_Direct',
               'distribution_channel_GDS', 'distribution_channel_TA/TO',
               'customer_type_Contract', 'customer_type_Transient',
               'customer_type_Transient-Party']

@app.post('/predict') 
def predict():
    data = request.json
    print(data)

    try:
        data = pd.DataFrame(data)
    except ValueError:
        data = pd.DataFrame([data])

    if list(data.columns) == covariables:
        try:
            sample = data.values
            print("Sample is", sample)
        except KeyError:
            return jsonify({'error':'Invalid input'})
        
        sample_scaled = scaler_fitted.transform(sample)

        predictions = predict_cancellation(sample_scaled)
        print(predictions)
        
        try:
            result = jsonify(predictions)
        except TypeError as e:
            return jsonify({'error':str(e)})
        
        return result
    
    else:
        return jsonify({'error':'Invalid input'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)