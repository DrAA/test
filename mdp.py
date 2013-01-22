from universe import end_of_time

def online_mdp(self, environment, model):
    t = 0
    while not end_of_time():
        observation = environment.get_new_data(t)
        prediction = model.predict(t, observation)
        loss = environment.calc_loss(t, prediction)
        model.update(prediction, loss)
        t += 1
