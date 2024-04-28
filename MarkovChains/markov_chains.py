# markov_chains.py
"""Volume 2: Markov Chains.
Sophia Carter

"""

import numpy as np
from scipy import linalg as la
import pandas as pd


class MarkovChain:
    """A Markov chain with finitely many states.
        
    Attributes:
        A: a column stochastic
        dictionary: mapping the state labels to the row/column index that
            they correspond to in A (given by order of the labels in the list)
        states: the states given
        

    """
    def __init__(self, A, states=None):
        """Check that A is column stochastic and construct a dictionary
        mapping a state's label to its index (the row / column of A that the
        state corresponds to). Save the transition matrix, the list of state
        labels, and the label-to-index dictionary as attributes.

        Parameters:
        A ((n,n) ndarray): the column-stochastic transition matrix for a
            Markov chain with n states.
        states (list(str)): a list of n labels corresponding to the n states.
            If not provided, the labels are the indices 0, 1, ..., n-1.

        Raises:
            ValueError: if A is not square or is not column stochastic.

        Example:
            >>> MarkovChain(np.array([[.5, .8], [.5, .2]], states=["A", "B"])
        corresponds to the Markov Chain with transition matrix
                                   from A  from B
                            to A [   .5      .8   ]
                            to B [   .5      .2   ]
        and the label-to-index dictionary is {"A":0, "B":1}.
        """
        #if not stochastic
        if not np.allclose(A.sum(axis=0),np.ones(A.shape[1])):
            raise ValueError("A is not column stochastic")

        #Attributes
        n,n =np.shape(A)
        if states is None:
            states = np.arange(n)

        self.dictionary = {}
        self.A = A
        self.states = states

        for i in range(0, n):
            self.dictionary.update({states[i]:i})



    def transition(self, state):
        """Transition to a new state by making a random draw from the outgoing
        probabilities of the state with the specified label.

        Parameters:
            state (str): the label for the current state.

        Returns:
            (str): the label of the state to transitioned to.
        """
        #find probablilty array for the multinomial probability
        val = int(self.dictionary[state])
        probability_array = np.array((self.A[:,val]))
        trans_array = np.random.multinomial(1, probability_array)

        #find the index where 1 is
        trans_state = self.states[np.argmax(trans_array)]

        return(trans_state)

        

    def walk(self, start, N):
        """Starting at the specified state, use the transition() method to
        transition from state to state N-1 times, recording the state label at
        each step.

        Parameters:
            start (str): The starting state label.

        Returns:
            (list(str)): A list of N state labels, including start.
        """
        #start at specified state 
        
        state_list = [start]
        #1st transition
        trans_state = self.transition(start)
        state_list.append(trans_state)
        i = 0
        #use prob 2 to transition from state to state N - 2 times 
        while i < N - 2:
            trans_state = self.transition(trans_state)
            state_list.append(trans_state)
            i += 1
       
        #return list of N state labels including initial state
        return state_list


    # Problem 3
    def path(self, start, stop):
        """Beginning at the start state, transition from state to state until
        arriving at the stop state, recording the state label at each step.

        Parameters:
            start (str): The starting state label.
            stop (str): The stopping state label.

        Returns:
            (list(str)): A list of state labels from start to stop.
        """
        
        state_list = [start]
        if start is stop:
            return state_list

        #1st transition
        trans_state = self.transition(start)
        state_list.append(trans_state)
        #use prob 2 to transition from state to state 
        while trans_state is not stop:
            trans_state = self.transition(trans_state)
            state_list.append(trans_state)

        return state_list
            
        

    # Problem 4
    def steady_state(self, tol=1e-12, maxiter=40):
        """Compute the steady state of the transition matrix A.

        Parameters:
            tol (float): The convergence tolerance.
            maxiter (int): The maximum number of iterations to compute.

        Returns:
            ((n,) ndarray): The steady state distribution vector of A.

        Raises:
            ValueError: if there is no convergence within maxiter iterations.
        """
        #generate a random state distribution vector x0 
        n, n = np.shape(self.A)
        x_0 = np.array([0.5])
        while np.sum(x_0) != 1.0:
            x_0 = np.random.dirichlet(np.ones(n), size=1)
        x_0 = np.ravel(x_0)
    
        #calculate x_k+1 = Ax_k until 1-norm(x_k-1 - x_k) < tol
        x_k1 = self.A @ x_0  
        norm_val = np.linalg.norm(x_0 - x_k1, 1)
        if norm_val < tol:
                return x_k1
        k=1
        while k < maxiter:
            #calculate x_k+1 = Ax_k until 1-norm(x_k-1 - x_k) < tol
            old_x = x_k1
            x_k1 = self.A @ old_x
            norm_val = np.linalg.norm(old_x - x_k1, 1)

            #return approx steady state distribution
            if norm_val < tol:
                return(x_k1)
            k += 1

        #if k exceeds maxiter raise valueError
        raise ValueError("A^k does not converge")
        

class SentenceGenerator(MarkovChain):
    """A Markov-based simulator for natural language.

    Attributes:
        A: a column stochastic
        dictionary: mapping the state labels to the row/column index that
            they correspond to in A (given by order of the labels in the list)
        states: the states given

        
    """
    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents. You may assume that the file has one complete sentence
        written on each line.
        """
        self.dictionary = {}
        def MakeTransitionMatrix(filename):
            word_list = []
            sentences = []
            #read file from filename
            with open(filename) as file:
                sentences = [line.split(" ") for line in file.readlines()]

                #get unique words in the training set (state labels)
                i = 0
                j = 0
                while i < len(sentences):
                    j = 0
                    while j < len(sentences[i]):
                        word_list.append(sentences[i][j])
                        j += 1
                    i += 1
            self.states = list(set(word_list))
            #Add labels "$tart" and "$top"
            self.states.append("$top")
            self.states.insert(0, "$tart")
                
            #initialize square matrix of zeros to be the transition matrix
            transition_matrix = np.zeros((len(self.states), len(self.states)))

            #note: sentences are already split
            for i in sentences:
                i.append("$top")
                i.insert(0,"$tart")
                
                # for each consecutive pair of (x,y) of words in the list of words
                for k in range(0,len(i) - 1):
                    # add 1 to the entry of the transition matrix that corresponds
                    # to transitioning from state x to state y
                    x = self.states.index(i[k])
                    y = self.states.index(i[k + 1])
                    transition_matrix[y, x] += 1

            # make $top map to $top
            transition_matrix[len(self.states) - 1, len(self.states) - 1] = 1
            # normalize
            transition_matrix = transition_matrix / (transition_matrix.sum(axis=0))
            self.A = transition_matrix
        MakeTransitionMatrix(filename)
        n,n =np.shape(self.A)
        for i in range(0, n):
            self.dictionary.update({self.states[i]:i})

    
    def babble(self):
        """Create a random sentence using MarkovChain.path().

        Returns:
            (str): A sentence generated with the transition matrix, not
                including the labels for the $tart and $top states.

        Example:
            >>> yoda = SentenceGenerator("yoda.txt")
            >>> print(yoda.babble())
            The dark side of loss is a path as one with you.
        """
        
        #generate a path from start state to the stop state
        talk = self.path("$tart", "$top")
        #remove $tart and $top
        talk = talk[1:-1]
        #and join
        talk = " ".join(talk)

        return talk


train = SentenceGenerator("ad_scripts.txt")

#create 10 "ad reads"
for i in range(10):
    #create an ad read
    ad_read = train.babble()
    #Put into it's own txt file
    with open(f"ad_reads_babble{i}.txt", "w") as file:
        file.write(ad_read)