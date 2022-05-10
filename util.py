import torch


# input: model, dataset, 
# output: accuracy

class Tester:
    def __init__(self, img_encoder, model, dataset, limit=20):
        self.model = model
        self.img_encoder = img_encoder
        self.dataset = dataset
        self.limit = limit  # maximum number of allowed steps
        

    def run(self):
        self.model.eval()
        correct = 0
        for task in self.dataset:
            mem_state = self.model.get_init_states()
            p_action = torch.zeros([1, 1, 4])
            for _ in range(self.limit):
                img = task.transform(task.img()).unsqueeze(0)
                img = self.img_encoder(img)
                logit, mem_state = self.model(img, p_action, mem_state)     
                logit = logit.flatten()
                action = logit.argmax().item()
                reward = task.step(action)
                if reward == 0:
                    correct += 1
                    break
                p_action = torch.eye(4)[action].view(1, 1, -1)
            task.reset()

        return correct / len(self.dataset)
        

   