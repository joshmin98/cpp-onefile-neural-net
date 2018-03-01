#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

struct Connection {
  double weight, deltaWeight;
};

class Neuron;

typedef std::vector<Neuron> Layer;

class Neuron {
public:
  Neuron(unsigned int numOutputs, unsigned myIndex);
  void setOutputVal(double val) { m_outputVal = val; }
  double getOutputVal() const { return m_outputVal; }
  void feedForward(const Layer &prevLayer);
  void calcOutputGradients(double targetVal);
  void calcHiddenGradients(const Layer &nextLayer);
  void updateInputWeights(Layer &prevLayer);

private:
  double sumDOW(const Layer &nextLayer) const;
  static double transferFunction(double x);
  static double transferFunctionDerivative(double x);
  static double randomWeight(void) { return rand() / (double)RAND_MAX; }

  static double eta, alpha;
  double m_outputVal, m_gradient;
  unsigned m_myIndex;
  std::vector<Connection> m_outputWeights;
};

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

Neuron::Neuron(unsigned numOutputs, unsigned myIndex) {
  for (unsigned c = 0; c < numOutputs; ++c) {
    m_outputWeights.push_back(Connection());
    m_outputWeights.back().weight = randomWeight();
  }
  m_myIndex = myIndex;
}

double Neuron::transferFunction(double x) {
  // Using hyperbolic tangent as our activation function
  return tanh(x);
}

double Neuron::transferFunctionDerivative(double x) {
  // tanh derivative approximation
  return (1.0 - (x * x));
}

void Neuron::feedForward(const Layer &prevLayer) {
  double sum = 0.0;

  for (unsigned n = 0; n < prevLayer.size(); ++n) {
    sum += prevLayer[n].m_outputVal *
           prevLayer[n].m_outputWeights[m_myIndex].weight;
  }
  m_outputVal = Neuron::transferFunction(sum);
}

void Neuron::calcOutputGradients(double targetVal) {
  double delta = targetVal - m_outputVal;
  m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::sumDOW(const Layer &nextLayer) const {
  double sum = 0.0;

  for (unsigned n = 0; n < nextLayer.size(); ++n) {
    sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
  }
  return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer) {
  double dow = sumDOW(nextLayer);
  m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::updateInputWeights(Layer &prevLayer) {
  for (unsigned n = 0; n < prevLayer.size(); ++n) {
    Neuron &neuron = prevLayer[n];
    double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
    double newDeltaWeight = eta //overall training rate
                            * neuron.getOutputVal()
                            * m_gradient
                            * alpha // "momentum" or a fraction of the previous delta weight
                            * oldDeltaWeight;
    neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
    neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
  }
}

class Net {
public:
  Net(const std::vector<unsigned int> &topology);
  void feedForward(const std::vector<double> &inputVals);
  void backProp(const std::vector<double> &targetVals);
  void getResults(std::vector<double> resultVals) const;

private:
  std::vector<Layer> m_layers;
  double m_error, m_recentAverageError, m_recentAverageSmoothingFactor;
};

Net::Net(const std::vector<unsigned int> &topology) {
  unsigned numLayers = topology.size();
  for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
    m_layers.push_back(Layer());
    unsigned numOutputs =
        layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
    for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
      m_layers.back().push_back(Neuron(numOutputs, neuronNum));
      std::cout << "neuron created" << std::endl;
    }
  }
}

void Net::feedForward(const std::vector<double> &inputVals) {
  assert(inputVals.size() == m_layers[0].size() - 1);

  for (unsigned i = 0; i < inputVals.size(); ++i) {
    m_layers[0][i].setOutputVal(inputVals[i]);
  }

  // Forward propogation
  for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
    Layer &prevLayer = m_layers[layerNum - 1];
    for (unsigned n = 0; n < m_layers[layerNum].size(); ++n) {
      m_layers[layerNum][n].feedForward(prevLayer);
    }
  }
}

void Net::backProp(const std::vector<double> &targetVals) {
  // Calculate overall net error (RMS)
  Layer &outputLayer = m_layers.back();
  m_error = 0.0;

  for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
    double delta = targetVals[n] - outputLayer[n].getOutputVal();
    m_error += delta * delta;
  }
  m_error /= outputLayer.size() - 1;
  m_error = sqrt(m_error);

  // Implement a recent average measurement
  m_recentAverageError =
      (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) /
      (m_recentAverageSmoothingFactor + 1.0);

  // Calculate output layer gradients
  for (unsigned n = 0; n < outputLayer.size(); ++n) {
    outputLayer[n].calcOutputGradients(targetVals[n]);
  }

  // Calculate gradients on hidden layers
  for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
    Layer &hiddenLayer = m_layers[layerNum];
    Layer &nextLayer = m_layers[layerNum + 1];

    for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
      hiddenLayer[n].calcHiddenGradients(nextLayer);
    }
  }

  // For all layers from outputs to first hidden layer, update connection
  // weights
  for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
    Layer &layer = m_layers[layerNum];
    Layer &prevLayer = m_layers[layerNum - 1];

    for (unsigned n = 0; n < layer.size(); ++n) {
      layer[n].updateInputWeights(prevLayer);
    }
  }
}

void Net::getResults(std::vector<double> resultVals) const {
  resultVals.clear();
  for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
    resultVals.push_back(m_layers.back()[n].getOutputVal());
  }
}

int main(int argc, char *argv[]) {
  std::vector<unsigned int> topology;
  topology.push_back(3);
  topology.push_back(2);
  topology.push_back(1);
  Net myNet(topology);

  std::vector<double> inputVals;
  myNet.feedForward(inputVals);

  std::vector<double> targetVals;
  myNet.backProp(targetVals);

  std::vector<double> resultVals;
  myNet.getResults(resultVals);
}
