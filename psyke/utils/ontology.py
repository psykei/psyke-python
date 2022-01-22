import numbers
from owlready2 import *
import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype, is_string_dtype
from tuprolog.theory import Theory


def dataframe_to_ontology(dataframe: pd.DataFrame) -> owlready2.Ontology:
    name = dataframe.columns[-1]
    ontology = get_ontology(f"{name[0].lower() + name[1:]}.rdf")
    with ontology:
        Main = type(name.capitalize(), (Thing,), {})
        for column in dataframe.columns:
            if is_numeric_dtype(dataframe[column]):
                type(column, (Main >> float, FunctionalProperty), {})
            elif is_string_dtype(dataframe[column]):
                type(column, (Main >> str, FunctionalProperty), {})
            else:
                raise TypeError
    return ontology


def individuals_from_dataframe(ontology: owlready2.Ontology, dataframe: pd.DataFrame) -> list:
    # owlready2.Ontology:
    # individuals = get_ontology(f"{ontology.name}_individuals.rdf")
    # individuals.imported_ontologies.append(ontology)
    Main = type(ontology.name.capitalize(), (Thing,), {})
    # with ontology:
    ret = []
    for _, row in dataframe.iterrows():
        string_row = "ontology.Iris("
        for name, value in zip(row.index, row):
            if isinstance(value, numbers.Number):
                string_row += f"{name}={value},"
            elif isinstance(value, str):
                string_row += f"{name}='{value}',"
            else:
                raise TypeError
        individual = eval(string_row[:-1] + ")")
        ret.append(individual)
        # destroy_entity(individual)
    # ret = list(individuals.individuals())
    # individuals.destroy()
    return ret


def rules_from_theory(ontology: owlready2.Ontology, theory: Theory, dataframe: pd.DataFrame) -> owlready2.Ontology:
    name = ontology.name
    # name = dataframe.columns[-1]
    lower_name = name[0].lower() + name[1:]
    Main = type(name.capitalize(), (Thing,), {})
    with ontology:
        Imp().set_as_rule(eval("Main").name +
                          f"""(?{lower_name}), PetalLength(?{lower_name}, ?pl), SepalLength(?{lower_name}, ?sl), 
            PetalWidth(?{lower_name}, ?pw), SepalWidth(?{lower_name}, ?sw), greaterThanOrEqual(?pl, 5) -> 
            {lower_name}(?{lower_name}, 'versicolor')""")
        Imp().set_as_rule(eval("Main").name +
                          f"""(?{lower_name}), PetalLength(?{lower_name}, ?pl), SepalLength(?{lower_name}, ?sl), 
                    PetalWidth(?{lower_name}, ?pw), SepalWidth(?{lower_name}, ?sw), lessThan(?pl, 5) -> 
                    {lower_name}(?{lower_name}, 'virginica')""")
        # rule2.set_as_rule(
        #     """Iris(?i), PetalLength(?i, ?pl), SepalLength(?i, ?sl), PetalWidth(?i, ?pw), SepalWidth(?i, ?sw),
        #     lessThan(?pl, 5) -> iris(?i, 'versicolor')""")
    # for c in theory:
    #    print(c.head)
    #    for h in c.head:
    #        print(h.name)
    #    print(c.body)
    #    break
    return ontology
