from abc import ABC, abstractmethod

class Launcher(ABC):
    @abstractmethod
    def launch_training(self):
        """Méthode que toutes les classes Laucher doivent implémenter"""
        pass

    @abstractmethod
    def launch_test(self):
        """Méthode que toutes les classes Laucher doivent implémenter"""
        pass
