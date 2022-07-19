import os
import sys
from datetime import datetime

import click
import pandas as pd
import dill


@click.command()
@click.option(
    '--path',
    '-p',
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help='Path to csv file with data to make predictions from.'
)
@click.option(
    '--batch/--batch-nocache',
    default=None,
    help="""
        Use this flag if your path is a csv file with batch of online learning data for the model.
        Automatically caches previous model. Use --batch-nocache to disable this feature.
    """
)
def main(path, batch):
    """
    Predict flight procedure types from csv data in PATH using trained PAC model
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

        model_path = os.path.join(application_path, 'flattern_model_pac.pkl')
        with open(model_path, 'rb') as model_file:
            model_pkg = dill.load(model_file)
            model = model_pkg['model']
            le = model_pkg['label_encoder']

    except Exception as e:
        print(f'Something went wrong while recovering model. Error message: {e}')
        return 1

    if batch is not None:
        batch_X = data.drop(columns=['ProcedureType'])
        batch_y = le.transform(data['ProcedureType'])

        batch_X = model.named_steps['prep_dropna'].transform(batch_X)
        batch_X = model.named_steps['prep_drop_idx_cols'].transform(batch_X)
        batch_X = model.named_steps['scaler'].transform(batch_X)

        new_model = model.named_steps['pac'].partial_fit(batch_X, batch_y)
        model_pkg['model'].named_steps['pac'] = new_model

        if batch:
            cache_path = f'{application_path}/flattern_cache/model_{datetime.now().strftime("%Y%m%d%H%M")}.pkl'
            os.makedirs(f'{application_path}/flattern_cache', exist_ok=True)

            os.replace(model_path, cache_path)

        model_pkg['metadata']['date'] = datetime.now()
        with open(model_path, 'wb') as model_file:
            dill.dump(
                {
                    'model': model_pkg['model'],
                    'label_encoder': le,
                    'metadata': model_pkg['metadata']
                }, model_file
            )

        return 0

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
        points_path = os.path.join(application_path, 'AirspaceData_SSRoute.csv')
        points = pd.read_csv(points_path, sep=';').set_index(['SidStarID', 'Number'])

        points.loc[points.ProcedureType != 0, 'ProcedureType'] = 0

        preds = res[res.ProcedureType != 'ordinary']
        preds.loc[preds.ProcedureType == 'veer', 'RouteL'] -= 1

        point_preds = preds.loc[preds.index.repeat(preds.RouteL)].drop(columns=['CatalogStId'])
        increment = point_preds.groupby(['SidStarID', 'Number', 'RouteL']).cumcount()
        point_preds.loc[:, 'Number'] += increment
        point_preds = point_preds.set_index(['SidStarID', 'Number'])

        procedure_type_mapper = {
            'veer': 1,
            'trombone': 3
        }
        point_preds.loc[:, 'ProcedureType'] = point_preds.ProcedureType.map(procedure_type_mapper)

        points.loc[point_preds.index, 'ProcedureType'] = point_preds['ProcedureType']
        points = points.reset_index()

        cols = ['SSRouteID', 'SidStarID', 'Number'] + list(points.columns[3:])
        points = points[cols]

    except Exception as e:
        print(
            f'Something went wrong while connecting predictions and route points. Error message: {e}'
        )
        return 1

    try:
        with open('AirspaceData_SSRoute_Predictions.csv', 'w') as points_file:
            points.to_csv(points_file, sep=';', index=False, quoting=1, line_terminator='\n')

        with open('empty_data.csv', 'w') as empty_file:
            empty.to_csv(empty_file, index=False, line_terminator='\n')

    except Exception as e:
        print(f'Something went wrong while saving predictions. Error message: {e}')
        return 1

    return 0


if __name__ == '__main__':
    main()
