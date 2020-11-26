import finansp.time_series as ts

import click
import pandas as pd

@click.group()
def cli():
    pass

@click.command()
@click.option( '-v', '--value_to_predict', default="open", help='Value to predict.', show_default=True )
@click.option( '-p', '--periods_to_predict', default=90, help='Days to predict.', show_default=True )
@click.option( '-i', '--interval_width', default=0.95, help='Interval width for prediction.', show_default=True )
@click.option( '-d', '--last_days', default=2200, help='Days used for predicting.', show_default=True )
@click.option( '-s', '--show_charts', is_flag=True, default=False, help='Show graphics.', show_default=True )
@click.option( '-o', '--output', default="output.csv", help='File to save the output.', show_default=True )
@click.argument( 'companies', nargs=-1, required=True )
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
    click.echo( f"{output} FILE CREATED.")

@click.command()
@click.option( '-v', '--value_to_predict', default="open", help='Value to predict.', show_default=True )
@click.option( '-p', '--periods_to_predict', default=90, help='Days to predict.', show_default=True )
@click.option( '-i', '--interval_width', default=0.95, help='Interval width for prediction.', show_default=True )
@click.option( '-d', '--last_days', default=2200, help='Days used for predicting.', show_default=True )
@click.option( '-s', '--show_charts', is_flag=True, default=False, help='Show graphics.', show_default=True )
@click.option( '-o', '--output', default="output.csv", help='File to save the output.', show_default=True )
@click.option( '-e', '--max_iter', default=1, help='Number of epochs to train models.', show_default=True )
@click.option( '-h', '--hidden_layers', type=click.STRING, default="10,5", help='Hidden layers', show_default=True )
@click.option( '-b', '--block_size', default=128, help='Batch size.', show_default=True )
@click.option( '-S', '--seed', default=1234, help='Seed.', show_default=True )
@click.option( '-V', '--val_split', default=0.2, help='Validation split', show_default=True )
@click.option( '-c', '--classifier_type', default="mlp", help='Classifier used to predict', show_default=True )
@click.option( '--max_depth', default=5, help='Max Depth for the selected Tree Algorithm', show_default=True )
@click.option( '--max_bins', default=10, help='Max Bin for the selected Tree Algorithm', show_default=True )
@click.option( '--min_instances_per_node', default=1, help='Min Instances Per Node for the selected Tree Algorithm', show_default=True )
@click.option( '--num_trees', default=20, help='Number of Trees for Random Forest', show_default=True )
@click.argument( 'companies', nargs=-1, required=True )
def should_I_buy( companies, max_iter, hidden_layers, block_size, seed, val_split, last_days, interval_width, value_to_predict, periods_to_predict, show_charts, output, \
    classifier_type, max_depth, max_bins, min_instances_per_node, num_trees ):
    """
        Should you buy or sell? 0 - WAIT, 1 - BUY, 2 - SELL

        This function tells you when to buy and sell, using `predict` function and a classifier
        ( MultiLayer Perceptron (MLP) by default), in the `predictions` column. You can change the number 
        of epochs to train your classifier, `max_iter`, the number of `hidden_layers`, the batch size
        ( `block_size` ) and the amount of data to be used in the validation set (`validation_split`).

        You can also select some Tree Algorithms to predict and change their parameters:\n
            - RandomForest (`rf`)
            - DecisionTree (`dt`)

        You can select one of these values to make the predictions:\n
            - "open"\n
            - "close"\n
            - "high"\n
            - "low"\n
            - "volume"\n
            - "unadjustedVolume"\n
            - "change"\n
            - "changePercent"\n
            - "vwap"\n
            - "changeOverTime"\n
    """
    # Array of Hidden Layers
    hidden_layers = [ int( hl.strip() ) for hl in hidden_layers.split(',')]
    
    pdf = ts.to_buy_or_not_to_buy( 
        companies=companies, 
        max_iter=max_iter,
        hidden_layers=hidden_layers, 
        blockSize=block_size,
        seed=seed,
        val_split=val_split, 
        value_to_predict=value_to_predict,
        periods_to_predict=periods_to_predict,
        interval_width=interval_width,
        show_charts=show_charts,
        number_of_days=last_days,
        classifier_type=classifier_type, 
        max_depth=max_depth,
        max_bins=max_bins, 
        min_instances_per_node=min_instances_per_node, 
        num_trees=num_trees
    )

    pdf.to_csv( output, index=False )
    click.echo( f"{output} FILE CREATED.")

if __name__ == '__main__':
    cli.add_command(predict)
    cli.add_command(should_I_buy)
    cli()
