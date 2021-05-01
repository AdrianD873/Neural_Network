#include "Network.h"

Network::Network(const std::vector<unsigned int>& topology)
{
	for (unsigned int i = 0; i < topology.size(); i++)
	{
		auto& numNeurons = topology[i];
		Layer layer;

		if (i == 0)
		{
			for (unsigned int j = 0; j < numNeurons; j++)
			{
				layer.push_back(Neuron{ 1, j });
			}
		}
		else
		{
			for (unsigned int j = 0; j < numNeurons; j++)
			{
				layer.push_back(Neuron{ topology[i - 1], j });
			}
		}
		layers.push_back(layer);
	}
}

void Network::feedForward(const std::vector<double> input)
{
	auto& inputLayer = layers[0];
	auto& outputLAyer = layers.back();
	assert(input.size() == inputLayer.size());

	//output of input layer
	for (unsigned i = 0; i < inputLayer.size(); i++)
	{
		inputLayer[i].output = input[i];
	}

	//output of all other layers
	for (unsigned i = 1; i < layers.size(); i++)
	{
		auto& prevLayer = layers[i - 1];
		auto& currentLayer = layers[i];
		for (unsigned j = 0; j < currentLayer.size(); j++)
		{
			currentLayer[j].feedForward(prevLayer);
		}
	}
}

void Network::backPropagation(const std::vector<double>& target)
{
	auto& outputLayer = layers.back();

	// calculate cost
	cost = 0.0;
	for (unsigned i = 0; i < outputLayer.size(); i++)
	{
		double delta = target[i] - outputLayer[i].output;
		cost += pow(delta, 2);
	}
	cost /= outputLayer.size();
	cost = sqrt(cost);

	// calculate output layer gradients
	for (unsigned i = 0; i < outputLayer.size(); i++)
	{
		outputLayer[i].calcOutputGradients(target[i]);
	}

	// calculate hidden layer gradients
	for (int i = layers.size() - 2; i > 0; i--)
	{
		auto& currentLayer = layers[i];
		auto& nextLayer = layers[i + 1];

		for (unsigned j = 0; j < currentLayer.size(); j++)
		{
			currentLayer[j].calcHiddenGradients(nextLayer);
		}
	}
}

void Network::updateWeights()
{
	for (unsigned i = 1; i < layers.size(); i++)
	{
		auto& currentLayer = layers[i];
		auto& previousLayer = layers[i - 1];

		for (unsigned j = 0; j < currentLayer.size(); j++)
		{
			currentLayer[j].updateInputWeights(previousLayer);
			currentLayer[j].updateBias();
		}
	}
}

std::vector<double> Network::getResults() const
{
	std::vector<double> results;
	for (unsigned i = 0; i < layers.back().size(); i++)
	{
		results.push_back(layers.back()[i].output);
	}
	return results;
}
