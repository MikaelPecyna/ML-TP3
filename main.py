from view.engine import Engine

if __name__ == "__main__":
    test_qlearning()
    #test_qdeeplearning()

    engine = Engine(width=4, height=4)  
    engine.run()
    pass

