import numpy as np

class ConsensusCommittee:
    def __init__(self, weights, agent_coeffs, threshold_auto_confirm=0.95, threshold_consensus_lower=0.6,
                 threshold_consensus_upper=0.8, threshold_pred_model=0.85):
        """
        Initialize the committee with weights and agent coefficients.

        :param weights: list of weights for chromatic, HOG, and depth models [w1, w2, w3]
        :param agent_coeffs: list of significance coefficients for each agent [A1, A2, A3]
        :param threshold_auto_confirm: threshold for automatic defect confirmation
        :param threshold_consensus_lower: lower bound threshold for consensus process
        :param threshold_consensus_upper: upper bound threshold for consensus process
        :param threshold_pred_model: threshold for probabilistic prediction model
        """
        self.weights = weights
        self.agent_coeffs = agent_coeffs
        self.threshold_auto_confirm = threshold_auto_confirm
        self.threshold_consensus_lower = threshold_consensus_lower
        self.threshold_consensus_upper = threshold_consensus_upper
        self.threshold_pred_model = threshold_pred_model

    def sigmoid(self, x):
        """Sigmoid function for probability scaling."""
        return 1 / (1 + np.exp(-x))

    def weighted_average(self, chromatic_prob, hog_prob, depth_prob):
        """
        Calculates the weighted average of defect probabilities using the provided formula:

        Pvoc = Q(Pdef) = (ω1 * A1 * Pdef1 + ω2 * A2 * Pdef2 + ω3 * A3 * Pdef3) / (ω1 + ω2 + ω3)

        :param chromatic_prob: probability from chromatic model
        :param hog_prob: probability from HOG model
        :param depth_prob: probability from depth map model
        :return: weighted average of probabilities
        """
        numerator = (self.weights[0] * self.agent_coeffs[0] * chromatic_prob +
                     self.weights[1] * self.agent_coeffs[1] * hog_prob +
                     self.weights[2] * self.agent_coeffs[2] * depth_prob)
        denominator = sum(self.weights)
        weighted_avg = numerator / denominator

        if isinstance(weighted_avg, np.ndarray):
            weighted_avg = np.mean(weighted_avg)
        return weighted_avg

    def evaluate(self, chromatic_prob, hog_prob, depth_prob, pred_prob=None):
        """
        Evaluate defect probability based on agent results using all provided thresholds.

        :param chromatic_prob: probability from chromatic model
        :param hog_prob: probability from HOG model
        :param depth_prob: probability from depth model
        :param pred_prob: probability from probabilistic prediction model (optional)
        :return: evaluation result and final probability
        """
        # Step 1: Calculate the weighted average
        weighted_avg = self.weighted_average(chromatic_prob, hog_prob, depth_prob)
        result = self.sigmoid(weighted_avg)

        # Step 2: Check automatic defect confirmation
        if result >= self.threshold_auto_confirm:
            return "Defect Confirmed", result

        # Step 3: Human needed due high k
        elif self.threshold_consensus_upper < result < self.threshold_auto_confirm:
            return "Human needed", result

        # Step 4: Check if consensus process is needed
        elif self.threshold_consensus_lower <= result <= self.threshold_consensus_upper:
            print("резулт до обработки", result)
            final_result = result * 0.22
            if final_result <= self.threshold_consensus_lower:
                return "Start Consensus Process", final_result
            else:
                return "Human needed", final_result

        # Step 5: Check if probabilistic prediction model should be used
        # elif result <= self.threshold_pred_model:
        #     if pred_prob is not None:
        #         # Apply formula for probabilistic prediction model integration:
        #         final_result = self.sigmoid(result + self.weights[3] * pred_prob)
        #         if final_result <= self.threshold_auto_confirm:
        #             return "Human Expert Needed", final_result
        #         else:
        #             return "Defect Confirmed with Prediction", final_result
        #     else:
        #         return "Human Expert Needed", result

        # Step 6: No defect detected
        return "No Defect", result
