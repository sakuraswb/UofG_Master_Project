def predict_masks(model, test_gen):
    preds = model.predict(test_gen)
    return preds
