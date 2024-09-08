# -*- coding: utf-8 -*-
import click
import logging
import shutil
import os

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    if os.path.exists(output_filepath):
        shutil.rmtree(output_filepath)
    
    shutil.copytree(input_filepath, output_filepath)
    
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
