import nn


class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        res = nn.as_scalar(nn.DotProduct(x, self.w))
        if res >= 0.0:
            return 1.0
        return -1.0

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        res = self.train_once(dataset)
        while res > 0.0:
            res = self.train_once(dataset)

    def train_once(self, dataset) -> float:
        """
        Train the perceptron once, return failure rate.
        """
        "*** YOUR CODE HERE ***"
        total = 0
        failure = 0
        for x, y in dataset.iterate_once(1):
            res = self.run(x)
            total += 1
            if nn.as_scalar(nn.DotProduct(y, res)) >= 0.0:
                continue
            # update the weight as to be.
            failure += 1
            if self.get_prediction(x) >= 0.0:
                # lower the score of false answer.
                self.w.update(x, -1.0)
            else:
                # raise score of correct answer.
                self.w.update(x, 1.0)
        return failure / total


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.hidden_size = 512
        self.learning_rate = 0.05
        self.batch_size = 200
        self.input = nn.Parameter(1, self.hidden_size)
        self.input_bias = nn.Parameter(1, self.hidden_size)
        self.hidden = nn.Parameter(self.hidden_size, 1)
        self.hidden_bias = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        input_res = nn.Linear(x, self.input)
        input_res = nn.AddBias(input_res, self.input_bias)
        input_res = nn.ReLU(input_res)
        res = nn.Linear(input_res, self.hidden)
        res = nn.AddBias(res, self.hidden_bias)
        return res

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predict_y = self.run(x)
        return nn.SquareLoss(predict_y, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while self.train_once(dataset) > 0.015:
            pass

    def train_once(self, dataset) -> float:
        """ return the average loss of one batch of examples"""
        total_loss = 0.0
        count = 0
        for x, y in dataset.iterate_once(self.batch_size):
            loss = self.get_loss(x, y)
            count += 1
            grad_input, grad_input_b, grad_hidden, grad_hidden_b = nn.gradients(loss,
                                                                                [self.input, self.input_bias,
                                                                                 self.hidden, self.hidden_bias])
            total_loss += loss.data.item()
            self.input.update(grad_input, -self.learning_rate)
            self.input_bias.update(grad_input_b, -self.learning_rate)
            self.hidden.update(grad_hidden, -self.learning_rate)
            self.hidden_bias.update(grad_hidden_b, -self.learning_rate)
        return total_loss / count


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.5
        self.batch_size = 100
        self.hidden_size = 200
        # self.num_class = 10
        self.input = nn.Parameter(784, self.hidden_size)
        self.input_bias = nn.Parameter(1, self.hidden_size)
        self.hidden = nn.Parameter(self.hidden_size, 10)
        self.hidden_bias = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        input_res = nn.Linear(x, self.input)
        input_res = nn.AddBias(input_res, self.input_bias)
        input_res = nn.ReLU(input_res)
        res = nn.Linear(input_res, self.hidden)
        res = nn.AddBias(res, self.hidden_bias)
        return res

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predict_y = self.run(x)
        return nn.SoftmaxLoss(predict_y, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while self.train_once(dataset) < 0.975:
            pass

    def train_once(self, dataset):
        """ return the accuracy of the model"""
        accuracy = []
        for x, y in dataset.iterate_once(self.batch_size):
            loss = self.get_loss(x, y)
            grad_input, grad_input_b, grad_hidden, grad_hidden_b = nn.gradients(loss,
                                                                                [self.input, self.input_bias,
                                                                                 self.hidden, self.hidden_bias])
            accuracy.append(dataset.get_validation_accuracy())
            self.input.update(grad_input, -self.learning_rate)
            self.input_bias.update(grad_input_b, -self.learning_rate)
            self.hidden.update(grad_hidden, -self.learning_rate)
            self.hidden_bias.update(grad_hidden_b, -self.learning_rate)
        return sum(accuracy) / len(accuracy)


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        self.batch_size = 100
        self.hidden_size = 200
        self.learning_rate = 0.25

        self.W_x = nn.Parameter(self.num_chars, self.hidden_size)
        self.W_hidden = nn.Parameter(self.hidden_size, self.hidden_size)
        self.output = nn.Parameter(self.hidden_size, 5)
        self.output_bias = nn.Parameter(1, 5)

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the initial (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # the first letter.
        res = nn.Linear(xs[0], self.W_x)
        res = nn.ReLU(res)
        for i in range(1, len(xs)):
            res = nn.Add(nn.Linear(xs[i], self.W_x), nn.Linear(res, self.W_hidden))
            res = nn.ReLU(res)
        res = nn.Linear(res, self.output)
        res = nn.AddBias(res, self.output_bias)
        return res

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predict_y = self.run(xs)
        return nn.SoftmaxLoss(predict_y, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while self.train_once(dataset) < 0.82:
            pass

    def train_once(self, dataset) -> float:
        accuracy = []
        for x, y in dataset.iterate_once(self.batch_size):
            loss = self.get_loss(x, y)
            grad_W_x, grad_W_h, grad_out, grad_out_b = nn.gradients(loss, [self.W_x, self.W_hidden,
                                                                           self.output, self.output_bias])
            self.W_x.update(grad_W_x, -self.learning_rate)
            self.W_hidden.update(grad_W_h, -self.learning_rate)
            self.output.update(grad_out, -self.learning_rate)
            self.output_bias.update(grad_out_b, -self.learning_rate)
            accuracy.append(dataset.get_validation_accuracy())
        return sum(accuracy) / len(accuracy)