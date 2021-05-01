#ifndef NEURON_H
#define NEURON_H

#include <vector>

class Neuron;
typedef std::vector<Neuron> Layer;

class Neuron
{
public:
	Neuron(unsigned int numInputs, unsigned int index);
	void feedForward(const Layer& prevLayer); // calculate output of the neuron
	void calcOutputGradients(long double targetVal);
	void calcHiddenGradients(const Layer& nextLayer);
	void updateInputWeights(Layer& prevLayer); // update all weights with the corresponding gradients
	void updateBias(); // update the bias with the current local gradient

	double output;

private:
	long double transferFunction(long double x); // tanh(x)
	long double transferFunctionDeriv(long double x); // 1.0 - x * x (approximation)
	double randomWeight() { return rand() / double(RAND_MAX); }

	const double eta = 0.15; // learning factor
	long double bias = 1.0;
	std::vector<long double> inputWeights;
	long double localGradient;
	unsigned int index; // "position" of the neuron in its layer
};

#endif // !NEURON_H