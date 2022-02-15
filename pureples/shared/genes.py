from neat.attributes import FloatAttribute, StringAttribute
from neat.genes import BaseGene


class OutputNodeGene(BaseGene):
    _gene_attributes = [FloatAttribute('obias'),
                        FloatAttribute('oresponse'),
                        StringAttribute('oactivation', options='sigmoid'),
                        StringAttribute('oaggregation', options='sum')]

    def __init__(self, key):
        assert isinstance(key, int), "OutputNodeGene key must be an int, not {!r}".format(key)
        BaseGene.__init__(self, key)

    @property
    def bias(self):
        return self.obias

    @bias.setter
    def bias(self, obias):
        self.obias = obias

    @property
    def response(self):
        return self.oresponse

    @response.setter
    def response(self, oresponse):
        self.oresponse = oresponse

    @property
    def activation(self):
        return self.oactivation

    @activation.setter
    def activation(self, oactivation):
        self.oactivation = oactivation

    @property
    def aggregation(self):
        return self.oaggregation

    @aggregation.setter
    def aggregation(self, oaggregation):
        self.oaggregation = oaggregation

    def distance(self, other, config):
        d = abs(self.obias - other.obias) + abs(self.oresponse - other.oresponse)
        if self.oactivation != other.oactivation:
            d += 1.0
        if self.oaggregation != other.oaggregation:
            d += 1.0
        return d * config.compatibility_weight_coefficient
