def fake_predict(age_input):
    if age_input > 10:
        prediction = "Survived (over 10)"
    else:
        prediction = "Super survived (under 10)"
    return prediction