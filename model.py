def nn_model(input_dim, output_dim=1, nodes=31):
	model = keras.Sequential()
	model.add(Dense(nodes, input_dim=input_dim, activation='relu'))
	model.add(Dense(nodes, activation='relu'))
	model.add(Dense(output_dim, activation='sigmoid'))
	optimiser = keras.optimizers.Adam(learning_rate = 0.0001) #default parameters used except for lr.
	model.compile(loss='categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])
	return model
