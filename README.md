# hotel-book-cancelation
Machine Learning application that predicts if a hotel booking, given some covariates values, is cancelled. The dataset used and some description can be found at https://www.sciencedirect.com/science/article/pii/S2352340918315191.

The application returns 0 (Not Cancelled) or 1 (Cancelled) via a POST request, that needs to be structured like this:
```
{  
    "lead_time": 142,
    "arrival_date_week_number": 17,
    "arrival_date_day_of_month": 22,
    "stays_in_weekend_nights": 2,
    "stays_in_week_nights": 3,
    "adults": 2,
    "babies": 0,
    "is_repeated_guest": 0,
    "previous_cancellations": 0,
    "previous_bookings_not_canceled": 0,
    "booking_changes": 0,
    "days_in_waiting_list": 0,
    "required_car_parking_spaces": 0,
    "total_of_special_requests": 0,
    "month": 1,
    "day": 18,
    "hotel_City Hotel": 1,
    "meal_BB": 1,
    "meal_HB": 0,
    "meal_SC": 0,
    "meal_Undefined": 0,
    "reserved_room_type_A": 1,
    "reserved_room_type_B": 0,
    "reserved_room_type_C": 0,
    "reserved_room_type_D": 0,
    "reserved_room_type_E": 0,
    "reserved_room_type_F": 0,
    "reserved_room_type_G": 0,
    "reserved_room_type_H": 0,
    "distribution_channel_Corporate": 0,
    "distribution_channel_Direct": 0,
    "distribution_channel_GDS": 0,
    "distribution_channel_TA/TO": 1,
    "customer_type_Contract": 0,
    "customer_type_Transient": 1,
    "customer_type_Transient-Party": 0
}
``` 

The final model is a neural network with two hidden layers and the output is computed via a sigmoid function. The data was standard scaled and the new sample to predict will be scaled also. Some data treatment and features importances analysis can be found at ml-dev/notebook.ipynb.

The application uses Docker and can be runned via the command (the image is published in my Docker Hub account):  
`docker run -p5000:5000 marcosaugusto47/hotel-book-app`