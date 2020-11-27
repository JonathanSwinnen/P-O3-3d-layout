import pickle

class DetectorPrerec():
    def __init__(self, filename):
        print(filename)
        file = open(filename, "rb")
        self.data = pickle.load(file)
        file.close()
    
    def detect_bot_frames(self, n):
        return self.data[n]
