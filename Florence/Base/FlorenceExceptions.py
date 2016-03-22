
class JacobianError(ArithmeticError):
    def __init__(self,value=None):
        self.value = value
    def __str__(self):
        if self.value is None:
            self.value = 'Jacobian of mapping is close to zero'
        return repr(self.value) 


class IllConditionedError(ArithmeticError):
    def __init__(self,value=None):
        self.value = value
    def __str__(self):
        if self.value is None:
            self.value = 'Matrix is ill conditioned'
        return repr(self.value)