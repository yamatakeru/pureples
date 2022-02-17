from neat.attributes import FloatAttribute, StringAttribute
from neat.genes import BaseGene


class OutputNodeGene(BaseGene):
    _gene_attributes = [FloatAttribute('output_bias'),
                        FloatAttribute('output_response'),
                        StringAttribute('output_activation', options='sigmoid'),
                        StringAttribute('output_aggregation', options='sum')]

    def __init__(self, key):
        assert isinstance(key, int), "OutputNodeGene key must be an int, not {!r}".format(key)
        BaseGene.__init__(self, key)

    @property
    def bias(self):
        return self.output_bias

    @bias.setter
    def bias(self, output_bias):
        self.output_bias = output_bias

    @property
    def response(self):
        return self.output_response

    @response.setter
    def response(self, output_response):
        self.output_response = output_response

    @property
    def activation(self):
        return self.output_activation

    @activation.setter
    def activation(self, output_activation):
        self.output_activation = output_activation

    @property
    def aggregation(self):
        return self.output_aggregation

    @aggregation.setter
    def aggregation(self, output_aggregation):
        self.output_aggregation = output_aggregation

    def distance(self, other, config):
        d = abs(self.output_bias - other.output_bias) + abs(self.output_response - other.output_response)
        if self.output_activation != other.output_activation:
            d += 1.0
        if self.output_aggregation != other.output_aggregation:
            d += 1.0

        return d * config.compatibility_weight_coefficient
