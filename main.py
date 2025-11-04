from view.engine import Engine
from core.qlearning import test_qlearning

if __name__ == "__main__":
    test_qlearning()
    #test_qdeeplearning()

    engine = Engine(width=4, height=4)  
    engine.run()
    pass

