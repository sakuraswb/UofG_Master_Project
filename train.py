from tensorflow.keras.callbacks import ModelCheckpoint

def train_model(model, train_gen, val_gen, save_path="model.h5", epochs=10):
    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
    callbacks = [ModelCheckpoint(save_path, save_best_only=True)]
    history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)
    return history
