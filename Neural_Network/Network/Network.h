#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <cassert>
#include "Neuron.h"

class Network
{
public:
	Network(const std::vector<unsigned int>& topology); // theoretical min topology: {1, 1}
	void feedForward(const std::vector<double> input); // calculate output of the network for a specific input
	void backPropagation(const std::vector<double>& target); // calculate all local gradients
	void updateWeights(); // update weights and biases
	std::vector<double> getResults() const; // get output vector of the network

	double cost; // loss of the last feed forward (calculated with back propagation)

private:
	std::vector<Layer> layers;

};

#endif // !NETWORK_H