from tensorflow.keras.callbacks import ModelCheckpoint

def train_model_ISIC(model, train_gen, val_gen, save_path="model_ISIC.h5", epochs=10):
    model.compile(
        optimizer="rmsprop",
        loss="binary_crossentropy",   
    )
    callbacks = [ModelCheckpoint(save_path, save_best_only=True)]
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks
    )
    return history
