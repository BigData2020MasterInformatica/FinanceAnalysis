import finansp.time_series as ts

import click
import pandas as pd

@click.command()
@click.option( '-v', '--value_to_predict', default="open", help='Value to predict.' )
@click.option( '-p', '--periods_to_predict', default=90, help='Days to predict.' )
@click.option( '-i', '--interval_width', default=0.95, help='Interval width for prediction.' )
@click.option( '-d', '--last_days', default=2200, help='Days used for predicting.' )
@click.option( '-s', '--show_charts', default=False, help='Show graphics.' )
@click.option( '-o', '--output', default="output.csv", help='File to save the output.' )
@click.argument('companies', nargs=-1, required=True)
def predict( companies, last_days, interval_width, value_to_predict, periods_to_predict, show_charts, output ):
    """
        Predict a the specified value, `value_to_predict`, for each company in the
        list `company_list` in the next `periods_to_predict` days, with an `interval_width`
        using the `last_days`. You can show graph by setting `show_charts` to True.

        Values to predict:\n
            - "open"\n
            - "close"\n
            - "high"\n
            - "low"\n
            - "close"\n
            - "volume"\n
            - "unadjustedVolume"\n
            - "change"\n
            - "changePercent"\n
            - "vwap"\n
            - "changeOverTime"\n
    """

    pdf = ts.predict( 
        company_list = companies,
        value_to_predict = value_to_predict,
        periods_to_predict = periods_to_predict,
        interval_width = interval_width,
        show_charts = show_charts, 
        number_of_days = last_days
    )

    pdf.to_csv( output, index=False )

if __name__ == '__main__':
    predict()
