import random
import pickle
import csv


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions


    def loadQ(self, filename):
        try:
            with open(filename + ".pickle", "rb") as f:
                self.q = pickle.load(f)  # Load the dict into self.q
                print("Loaded Q-table from file: {}".format(filename + ".pickle"))
        except FileNotFoundError:
            print("Error: File {}.pickle not found.".format(filename))
        except Exception as e:
            print("Error loading file:", e)
            

        print("Loaded file: {}".format(filename+".pickle"))


    def saveQ(self, filename):
        '''Save the Q state-action values to both a pickle file and a CSV file.'''
        # Save to pickle file
        try:
            with open(filename + ".pickle", "wb") as f:
                pickle.dump(self.q, f)  # Serialize the Q-table
                print("Wrote to pickle file: {}".format(filename + ".pickle"))
        except Exception as e:
            print("Error saving to pickle file:", e)

        # Save to CSV file
    try:
        with open(filename + ".csv", "w", newline="") as f:
            writer = csv.writer(f)
            # Write headers first
            writer.writerow(["State"] + self.actions)
            
            # Write Q-values for each state
            for state, action_values in self.q.items():
                row = [state]
                # Add Q-values for each action in a consistent order
                for action in self.actions:
                    row.append(action_values.get(action, 0))
                writer.writerow(row)
            
            print(f"Wrote to CSV file: {filename}.csv")
    except Exception as e:
        print("Error saving to CSV file:", e)



    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        
        # Initialize Q-values for this state if not existing
        for action in self.actions:
            if (state, action) not in self.q:
                self.q[(state, action)] = 0.0
        
        # Get Q-values for all actions in this state
        q_values = {action: self.q.get((state, action), 0.0) for action in self.actions}
        
        # Exploration (random action)
        if random.uniform(0, 1) < self.epsilon:
            act = random.choice(self.actions)
        else:
            # Check if all values are the same
            first_value = next(iter(q_values.values()))
            all_same = all(q == first_value for q in q_values.values())
            
            if all_same:
                # If all values are the same, choose completely randomly
                act = random.choice(self.actions)
            else:
                # Exploitation - choose action with highest Q-value
                max_q = max(q_values.values())
                best_actions = [action for action, q in q_values.items() if q == max_q]
                act = random.choice(best_actions)
        
        if return_q:
            return act, self.q
        return act
        
        """
        for action in self.actions:
            if (state, action) not in self.q:
                self.q[(state,action)] = 0.0


        choice = random.uniform(0,1)
        s1 = self.q[(state, self.actions[0])]
        s2 = self.q[(state, self.actions[1])]
        s3 = self.q[(state, self.actions[2])]
        if choice > self.epsilon or (s1 == s2 and s3 == s2):
            act = random.randint(0,2)
            if return_q:
                return self.actions[act], self.q
            return self.actions[act]
        else:
            if s1 == s2 and s3 < s2:
                act = random.randint(0,1)
            elif s3 == s2 and s1 < s2:
                act = random.randint(1,2)
            elif s3 == s1 and s2 < s1:
                act = random.randint(0,1)
                if(act == 0):
                    act = 2
            else:
                act = max(s1,max(s2,s3))
            if return_q:
                return self.actions[act], self.q
            return self.actions[act]
        """


        
        
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        '''
        # TODO: Implement exploration vs exploitation
        #    if we need to take a random action:
        #       * return a random action
        #    else:
        #       * determine which action has the highest Q value for the state 
        #          we are in.
        #       * address edge cases - what if 2 actions have the same max Q 
        #          value?
        #       * return the action with highest Q value
        #
        # NOTE: if return_q is set to True return (action, q) instead of
        #       just action


    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation
            '''

        # Initialize Q-values for all actions in both states if not existing
        for state in [state1, state2]:
            for action in self.actions:
                if (state, action) not in self.q:
                    self.q[(state, action)] = 0.0
        
        # Find the maximum Q-value for the next state
        next_state_q_values = {action: self.q[(state2, action)] for action in self.actions}
        
        # Check if all values are the same
        first_value = next(iter(next_state_q_values.values()))
        all_same = all(q == first_value for q in next_state_q_values.values())
        
        if all_same:
            # If all Q-values are identical, choose a random action for max_next_q
            chosen_action = random.choice(self.actions)
            max_next_q = next_state_q_values[chosen_action]
        else:
            # Otherwise use the maximum Q-value
            max_next_q = max(next_state_q_values.values())
        
        # Update Q-value using the Bellman equation
        # Q(s,a) += α * [R + γ * max(Q(s')) - Q(s,a)]
        self.q[(state1, action1)] += self.alpha * (
            reward + self.gamma * max_next_q - self.q[(state1, action1)]
        )


        """
        for action in self.actions:
            if (state1, action) not in self.q:
                self.q[(state1,action)] = 0.0

        for action in self.actions:
            if (state2, action) not in self.q:
                self.q[(state2, action)] = 0.0

        s1 = self.q[(state2, self.actions[0])]
        s2 = self.q[(state2, self.actions[1])]
        s3 = self.q[(state2, self.actions[2])]

        if s1 == s2 and s3 < s2:
            act = random.randint(0,1)
        elif s3 == s2 and s1 < s2:
            act = random.randint(1,2)
        elif s3 == s1 and s2 < s1:
            act = random.randint(0,1)
            if(act == 0):
                act = 2
        elif s1 == s2 and s2 == s3:
            act = self.actions[random.randint(0,2)]
        else:
            act = max(s1,max(s2,s3))


        self.q[(state1,action1)] += self.alpha*(reward + self.gamma * (self.q[(state2,self.actions[act])] - self.q[state1,action1]))

        """
        # TODO: Implement the Bellman update function:
        #     Q(s1, a1) += alpha * [reward(s1,a1) + gamma* max(Q(s2)) - Q(s1,a1)]
        # 
        # NOTE: address edge cases: i.e. 
        # 
        # Find Q for current (state1, action1)
        # Address edge cases what do we want to do if the [state, action]
        #       is not in our dictionary?
        # Find max(Q) for state2
        # Update Q for (state1, action1) (use discount factor gamma for future 
        #   rewards)

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE
