import os
import sys

import click
import pandas as pd
import dill


@click.command()
@click.option(
    '--path',
    '-p',
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help='Path to csv file with data to make predictions from'
)
def main(path):
    """
    Predict flight procedure types from csv data in PATH using trained decision tree model
    """

    pd.options.mode.chained_assignment = None

    data = pd.read_csv(path)

    try:
        # determine if application is a script file or frozen exe
        if getattr(sys, 'frozen', False):
            application_path = os.path.dirname(sys.executable)
        elif __file__:
            application_path = os.path.dirname(__file__)
        else:
            raise FileNotFoundError('Unable to resolve application path.')

        model_path = os.path.join(application_path, 'flattern_model_tree.pkl')
        with open(model_path, 'rb') as model_file:
            model_pkg = dill.load(model_file)
            model = model_pkg['model']
            le = model_pkg['label_encoder']

    except Exception as e:
        print(f'Something went wrong while recovering model. Error message: {e}')
        return 1

    try:
        index_columns = ['CatalogStId', 'SidStarID', 'Number', 'RouteL']
        res = data.dropna()[index_columns]

        predictions = le.inverse_transform(model.predict(data))

        res.loc[:, 'ProcedureType'] = predictions

        empty = data[data.isna().any(axis=1)][index_columns]

    except Exception as e:
        print(
            f'Something went wrong while making predictions. Please check input file. Error message: {e}'
        )
        return 1

    try:
        with open('predictions.csv', 'w') as out_file:
            res.to_csv(out_file, index=False, line_terminator='\n')

        with open('empty_data.csv', 'w') as empty_file:
            empty.to_csv(empty_file, index=False, line_terminator='\n')

    except Exception as e:
        print(f'Something went wrong while saving predictions. Error message: {e}')
        return 1

    return 0


if __name__ == '__main__':
    main()
