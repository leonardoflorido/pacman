from train import train
from utils import render, view

if __name__ == '__main__':
    agent = train()
    render(agent, 'MsPacmanDeterministic-v0')
    view()