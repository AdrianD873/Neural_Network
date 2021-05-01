#include "Neuron.h"

Neuron::Neuron(unsigned int numInputs, unsigned int index)
{
	for (unsigned i = 0; i < numInputs; i++)
	{
		inputWeights.push_back(randomWeight());
	}
	this->index = index;
}

void Neuron::feedForward(const Layer& prevLayer)
{
	double sum = 0.0;
	// calculate dot product
	for (unsigned i = 0; i < prevLayer.size(); i++)
	{
		sum += prevLayer[i].output * inputWeights[i];
	}
	
	output = transferFunction(sum + bias);
}

void Neuron::calcOutputGradients(long double targetVal)
{
	double delta = targetVal - output;
	localGradient = delta * transferFunctionDeriv(output);
}

void Neuron::calcHiddenGradients(const Layer& nextLayer)
{
	//gradients of each connection of the output
	double sum = 0.0;
	for (unsigned i = 0; i < nextLayer.size(); i++)
	{
		sum += nextLayer[i].inputWeights[index] * nextLayer[i].localGradient;
	}
	
	localGradient = sum * transferFunctionDeriv(output);
}

void Neuron::updateInputWeights(Layer& prevLayer)
{
	for (unsigned i = 0; i < prevLayer.size(); i++)
	{

		long double deltaWeight = eta * prevLayer[i].output * localGradient;
		inputWeights[i] += deltaWeight;
	}
}

void Neuron::updateBias() 
{
	bias *= localGradient;
}


long double Neuron::transferFunction(long double x)
{
	return tanh(x);
}

long double Neuron::transferFunctionDeriv(long double x)
{
	return 1.0 - x * x;
}
