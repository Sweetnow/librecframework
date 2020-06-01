#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, List, Dict, Any, Optional
from pathlib import Path
from abc import ABC, abstractmethod
import json
from argparse import ArgumentParser
import logging
import setproctitle
import torch
from torch.utils.data import DataLoader
from .argument.manager import HyperparamManager, \
    default_loader_argument_manager, default_env_argument_manager
from .data import DatasetFuncs
from .data.dataset import TrainDataset, FullyRankingTestDataset, LeaveOneOutTestDataset
from .utils.convert import name_to_metric, path_to_saved_model
from .utils.visshow import VisShow
from .utils.training import early_stop, check_overfitting
from .utils.gpu_selector import autoselect
from .model import Model
from .logger import Logger
from .train import train
from .test import fully_ranking_test, leave_one_out_test
from .metric import _Metric
from .trainhook import TrainHook, ValueMeanHook

# pylint: disable=W0221

__all__ = ['Pipeline',
           'DefaultFullyRankingTrainPipeline', 'DefaultLeaveOneOutTrainPipeline',
           'DefaultFullyRankingTestPipeline', 'DefaultLeaveOneOutTestPipeline',
           'DefaultFullyRankingPipeline', 'DefaultLeaveOneOutPipeline']


def _load_other_args(path: Union[str, Path]) -> dict:
    with open(path, 'r') as f:
        other_arg = json.load(f)
    return other_arg


class Pipeline(ABC):
    @abstractmethod
    def add_args(self, parser: ArgumentParser):
        return

    @abstractmethod
    def parse_args(self):
        return

    @abstractmethod
    def before_running(self, *args, **kwargs):
        return

    @abstractmethod
    def during_running(self, *args, **kwargs):
        return

    @abstractmethod
    def after_running(self, *args, **kwargs):
        return


class _DefaultTrainPipeline(Pipeline):
    def __init__(
            self,
            description: str,
            supported_datasets: List[str],
            train_funcs: DatasetFuncs,
            test_funcs: DatasetFuncs,
            hyperparam_manager: HyperparamManager,
            other_arg_path: Union[str, Path],
            pretrain_path: Union[str, Path],
            sample_tag: str,
            pin_memory: bool,
            min_memory: float,
            test_batch_size: int,
            # ==== partial parameters ====
            test_dataset_type: type,
            test_function):
        self._datasets = supported_datasets
        self._train_funcs = train_funcs
        self._test_funcs = test_funcs
        self._eam = default_env_argument_manager('both', self._datasets)
        self._lam = default_loader_argument_manager('both', test_batch_size)
        self._hpm = hyperparam_manager
        self._other_arg_path = other_arg_path
        self._pretrain_path = pretrain_path
        self._args, self._oargs, self._pretrain = None, None, None
        self._sample_tag = sample_tag

        self.description = description
        self.task = 'tune'
        self.metrics = None
        self.target_metric = None
        self.train_data, self.test_data = None, None
        self.train_loader, self.test_loader = None, None
        self.log = None
        self.pin_memory = pin_memory
        self.min_memory = min_memory

        self.infos = None
        self.model_class = None

        self._test_dataset_type = test_dataset_type
        self._test = test_function

        self.trainhooks = {
            '__loss__': ValueMeanHook('__loss__')
        }

    def __repr__(self):
        return f'{self.__class__.__name__}({self.description})'

    def add_args(self, parser: ArgumentParser):
        (self._eam + self._lam + self._hpm).add_args(parser)

    def parse_args(self):
        parser = ArgumentParser(description=self.description)
        self.add_args(parser)
        parser.parse_args()

    def before_running(self):
        # get args
        logging.basicConfig(level=logging.DEBUG)
        logging.info(self._eam)
        logging.info(self._lam)
        logging.info(self._hpm)

        self._oargs = _load_other_args(self._other_arg_path)
        self._pretrain = _load_other_args(self._pretrain_path)
        logging.info(self._oargs)
        logging.info(self._pretrain)

        # metrics
        metrics_args = self._oargs['metric']
        self.metrics = [name_to_metric(m['type'], m['topk'])
                        for m in metrics_args['metrics']]
        self.target_metric = name_to_metric(
            metrics_args['target']['type'],
            metrics_args['target']['topk'])

        # dataset
        dataset_args = self._oargs['dataset']
        self.train_data = TrainDataset(
            path=dataset_args['path'],
            name=self._eam['dataset'],
            num_worker=self._eam['sample_worker'],
            max_epoch=self._eam['sample_epoch'],
            funcs=self._train_funcs,
            sample_tag=self._sample_tag,
            seed=dataset_args['seed'],
            use_backup=dataset_args['use_backup']
        )
        self.test_data = self._test_dataset_type(
            path=dataset_args['path'],
            name=self._eam['dataset'],
            task=self.task,
            train_dataset=self.train_data,
            funcs=self._test_funcs
        )
        self.train_loader = DataLoader(
            self.train_data,
            batch_size=self._lam['batch_size'],
            shuffle=True,
            num_workers=self._lam['batch_worker'],
            pin_memory=self.pin_memory
        )
        self.test_loader = DataLoader(
            self.test_data,
            batch_size=self._lam['test_batch_size'],
            shuffle=False,
            num_workers=self._lam['test_batch_worker'],
            pin_memory=self.pin_memory
        )

        self.infos = self._hpm.to_infos()

    def during_running(
        self,
        model_class: Model,
        other_args: Dict[str, Any],
        trainhooks: Optional[Dict[str, TrainHook]] = None,
        optim_type = torch.optim.Adam):
        '''
        `other_args` will be sended to `model.__init__` as key-value args
        '''
        if trainhooks is None:
            trainhooks = self.trainhooks
        else:
            trainhooks.update(self.trainhooks)

        self.model_class = model_class
        modelname = self.model_class.__name__
        target = str(self.target_metric)

        logger_args = self._oargs['logger']
        logpath = Path(logger_args['path']) / modelname / \
            f"{self._eam['dataset']}_{self.task}"/self._eam['tag']
        self.log = Logger(
            logpath, logger_args['policy'],
            checkpoint_target=target)
        pretrain = 'pretrain' in self._hpm and self._hpm['pretrain']
        gpu_id = autoselect(self._eam['device'], self.min_memory)
        with torch.cuda.device(gpu_id):
            for info in self.infos:
                envdir = modelname
                sub_env = f"{self._eam['tag']}-{'-'.join(map(str,info._asdict().values()))}-{pretrain}"

                vis_args = self._oargs['visdom']
                vis = VisShow(
                    server=vis_args['server'],
                    port=vis_args['port'][self._eam['dataset']],
                    envdir=envdir,
                    subenv=sub_env
                )
                setproctitle.setproctitle(
                    f"{envdir}_{sub_env}_{self.log.random}@{self._oargs['user']}")

                model = self.model_class(
                    info, self.train_data, **other_args)
                if pretrain:
                    model.load_pretrain(self._pretrain[self._eam['dataset']])
                model = model.cuda()
                # op
                op = optim_type(model.parameters(), lr=info.lr)
                # env
                train_args = self._oargs['training']
                env = {
                    'op': str(op).split(' ')[0],
                    'dataset': self._eam['dataset'],
                    'training': train_args,
                    'target': target
                }
                logging.warning(f'Run {envdir}-{sub_env}')
                # log
                with self.log.update_modelinfo(modelname, info, env, target) as log:
                    # train
                    early = train_args['early_stop']
                    self.train_loader.dataset.init_epoch()
                    for epoch in range(self._eam['epoch']):
                        train(model, epoch + 1, self.train_loader, op, trainhooks)
                        for v in trainhooks.values():
                            vis.update(v.title, [epoch], [v.value])
                        log.update_trainhooks(trainhooks)
                        if epoch % train_args['test_interval'] != 0:
                            continue

                        self._test(model, self.test_loader, self.metrics)
                        for metric in self.metrics:
                            vis.update(str(metric), [epoch], [metric.metric])
                        log.update_metrics_and_model(self.metrics, model)

                        if epoch > train_args['overfit']['protected_epoch']:
                            if check_overfitting(
                                    log.metrics_log,
                                    target,
                                    train_args['overfit']['threshold']):
                                break

                        early = early_stop(
                            log.metrics_log[target],
                            early,
                            threshold=0)
                        if early <= 0:
                            break

    def after_running(self):
        pass


class DefaultFullyRankingTrainPipeline(_DefaultTrainPipeline):
    def __init__(
            self,
            description: str,
            supported_datasets: List[str],
            train_funcs: DatasetFuncs,
            test_funcs: DatasetFuncs,
            hyperparam_manager: HyperparamManager,
            other_arg_path: Union[str, Path],
            pretrain_path: Union[str, Path],
            sample_tag: str,
            pin_memory: bool,
            min_memory: float,
            test_batch_size: int = 4096):
        super().__init__(
            description,
            supported_datasets,
            train_funcs,
            test_funcs,
            hyperparam_manager,
            other_arg_path,
            pretrain_path,
            sample_tag,
            pin_memory,
            min_memory,
            test_batch_size,
            FullyRankingTestDataset,
            fully_ranking_test
        )


class DefaultLeaveOneOutTrainPipeline(_DefaultTrainPipeline):
    def __init__(
            self,
            description: str,
            supported_datasets: List[str],
            train_funcs: DatasetFuncs,
            test_funcs: DatasetFuncs,
            hyperparam_manager: HyperparamManager,
            other_arg_path: Union[str, Path],
            pretrain_path: Union[str, Path],
            sample_tag: str,
            pin_memory: bool,
            min_memory: float,
            test_batch_size: int = 4096):
        super().__init__(
            description,
            supported_datasets,
            train_funcs,
            test_funcs,
            hyperparam_manager,
            other_arg_path,
            pretrain_path,
            sample_tag,
            pin_memory,
            min_memory,
            test_batch_size,
            LeaveOneOutTestDataset,
            leave_one_out_test
        )


class _DefaultTestPipeline(Pipeline):
    def __init__(
            self,
            description: str,
            train_funcs: DatasetFuncs,
            test_funcs: DatasetFuncs,
            other_arg_path: Union[str, Path],
            pin_memory: bool,
            min_memory: float,
            test_batch_size: int,
            # ==== partial parameters ====
            test_dataset_type: type,
            test_function
    ):
        self._train_funcs = train_funcs
        self._test_funcs = test_funcs
        self._eam = default_env_argument_manager('test')
        self._lam = default_loader_argument_manager('test', test_batch_size)
        self._other_arg_path = other_arg_path

        self._args, self._oargs, self._pretrain = None, None, None

        self.description = description
        self.task = 'test'
        self.metrics = None
        self.target_metric = None
        self.dataset = None
        self.train_data = None
        self.test_data, self.test_loader = None, None
        self.log = None
        self.infos = None
        self.model_class = None
        self.pin_memory = pin_memory
        self.min_memory = min_memory

        self._test_dataset_type = test_dataset_type
        self._test = test_function

    def __repr__(self):
        return f'{self.__class__.__name__}({self.description})'

    def add_args(self, parser: ArgumentParser):
        (self._eam + self._lam).add_args(parser)

    def parse_args(self):
        parser = ArgumentParser(description=self.description)
        self.add_args(parser)
        parser.parse_args()

    def before_running(self):
        # get args
        logging.basicConfig(level=logging.DEBUG)
        logging.info(self._eam)
        logging.info(self._lam)

        self._oargs = _load_other_args(self._other_arg_path)
        logging.info(self._oargs)

        # metrics
        metrics_args = self._oargs['metric']
        self.metrics = [name_to_metric(m['type'], m['topk'])
                        for m in metrics_args['metrics']]
        self.target_metric = name_to_metric(
            metrics_args['target']['type'],
            metrics_args['target']['topk'])

        self.infos, self.dataset = path_to_saved_model(self._eam['file'])

        # dataset
        dataset_args = self._oargs['dataset']
        self.train_data = TrainDataset(
            path=dataset_args['path'],
            name=self.dataset,
            num_worker=0,
            max_epoch=0,
            funcs=self._train_funcs,
            sample_tag=None,
            seed=None,
            use_backup=False
        )
        self.test_data = self._test_dataset_type(
            path=dataset_args['path'],
            name=self.dataset,
            task=self.task,
            train_dataset=self.train_data,
            funcs=self._test_funcs
        )
        self.test_loader = DataLoader(
            self.test_data,
            batch_size=self._lam['test_batch_size'],
            shuffle=False,
            num_workers=self._lam['test_batch_worker'],
            pin_memory=self.pin_memory
        )

    def during_running(self, model_class: Model, other_args: Dict[str, Any]):
        self.model_class = model_class
        modelname = self.model_class.__name__
        target = str(self.target_metric)

        logger_args = self._oargs['logger']
        logpath = Path(logger_args['path']) / modelname / \
            f"{self.dataset}_{self.task}"/self._eam['tag']
        self.log = Logger(
            logpath, logger_args['policy'],
            checkpoint_target=target)
        gpu_id = autoselect(self._eam['device'], self.min_memory)
        with torch.cuda.device(gpu_id):
            for metadata in self.infos:
                eval_str = f"Eval: {modelname}_{metadata['id']}"
                setproctitle.setproctitle(
                    f"{eval_str}@{self._oargs['user']}")
                logging.warning(eval_str)

                info = metadata['info']
                model = self.model_class(
                    info, self.train_data, **other_args).cuda()
                model.load_state_dict(torch.load(
                    metadata['path'], map_location='cpu'))
                # log
                env = metadata['env']
                env['from'] = str(metadata['path'])
                with self.log.update_modelinfo(modelname, info, env, target) as log:
                    self._test(model, self.test_loader, self.metrics)
                    log.update_metrics_and_model(self.metrics, model)

    def after_running(self):
        pass


class DefaultFullyRankingTestPipeline(_DefaultTestPipeline):
    def __init__(
            self,
            description: str,
            train_funcs: DatasetFuncs,
            test_funcs: DatasetFuncs,
            other_arg_path: Union[str, Path],
            pin_memory: bool,
            min_memory: float,
            test_batch_size: int = 4096):
        super().__init__(
            description,
            train_funcs,
            test_funcs,
            other_arg_path,
            pin_memory,
            min_memory,
            test_batch_size,
            FullyRankingTestDataset,
            fully_ranking_test
        )


class DefaultLeaveOneOutTestPipeline(_DefaultTestPipeline):
    def __init__(
            self,
            description: str,
            train_funcs: DatasetFuncs,
            test_funcs: DatasetFuncs,
            other_arg_path: Union[str, Path],
            pin_memory: bool,
            min_memory: float,
            test_batch_size: int = 4096):
        super().__init__(
            description,
            train_funcs,
            test_funcs,
            other_arg_path,
            pin_memory,
            min_memory,
            test_batch_size,
            LeaveOneOutTestDataset,
            leave_one_out_test
        )


class _DefaultPipeline(Pipeline):
    def __init__(
            self,
            description: str,
            supported_datasets: List[str],
            train_funcs: DatasetFuncs,
            test_funcs: DatasetFuncs,
            hyperparam_manager: HyperparamManager,
            other_arg_path: Union[str, Path],
            pretrain_path: Union[str, Path],
            sample_tag: str,
            pin_memory: bool,
            min_memory: float,
            test_batch_size: int,
            # ==== partial parameters ====
            train_pipeline_type,
            test_pipeline_type
    ):
        self._train_pipeline = train_pipeline_type(
            description,
            supported_datasets,
            train_funcs,
            test_funcs,
            hyperparam_manager,
            other_arg_path,
            pretrain_path,
            sample_tag,
            pin_memory,
            min_memory,
            test_batch_size
        )
        self._test_pipeline = test_pipeline_type(
            description,
            train_funcs,
            test_funcs,
            other_arg_path,
            pin_memory,
            min_memory,
            test_batch_size
        )
        self.description = description
        self.which = None

    @property
    def train_data(self) -> TrainDataset:
        if self.which == 'train':
            return self._train_pipeline.train_data
        elif self.which == 'test':
            return self._test_pipeline.train_data
        else:
            raise RuntimeError(f'Unexpected `which` {self.which}')

    def __repr__(self):
        return f'{self.__class__.__name__}({self.description})'

    def add_args(self, parser: ArgumentParser):
        subparser = parser.add_subparsers(
            title='Pipelines',
            description='Choose training pipeline or testing pipeline')

        train_parser = subparser.add_parser(name='train')
        self._train_pipeline.add_args(train_parser)
        train_parser.set_defaults(which='train')

        test_parser = subparser.add_parser(name='test')
        self._test_pipeline.add_args(test_parser)
        test_parser.set_defaults(which='test')

    def parse_args(self):
        parser = ArgumentParser(description=self.description)
        self.add_args(parser)
        args = parser.parse_args()
        self.which = args.which

    def before_running(self):
        if self.which == 'train':
            self._train_pipeline.before_running()
        elif self.which == 'test':
            self._test_pipeline.before_running()
        else:
            raise RuntimeError(f'Unexpected `which` {self.which}')

    def during_running(
        self,
        model_class: Model,
        other_args: Dict[str, Any],
        trainhooks: Optional[Dict[str, TrainHook]] = None,
        optim_type = torch.optim.Adam):
        '''
        `other_args` will be sended to `model.__init__` as key-value args
        '''
        if self.which == 'train':
            self._train_pipeline.during_running(
                model_class, other_args, trainhooks, optim_type)
        elif self.which == 'test':
            self._test_pipeline.during_running(model_class, other_args)
        else:
            raise RuntimeError(f'Unexpected `which` {self.which}')

    def after_running(self):
        if self.which == 'train':
            self._train_pipeline.after_running()
        elif self.which == 'test':
            self._test_pipeline.after_running()
        else:
            raise RuntimeError(f'Unexpected `which` {self.which}')


class DefaultFullyRankingPipeline(_DefaultPipeline):
    def __init__(
            self,
            description: str,
            supported_datasets: List[str],
            train_funcs: DatasetFuncs,
            test_funcs: DatasetFuncs,
            hyperparam_manager: HyperparamManager,
            other_arg_path: Union[str, Path],
            pretrain_path: Union[str, Path],
            sample_tag: str,
            pin_memory: bool,
            min_memory: float,
            test_batch_size: int = 4096):
        super().__init__(
            description,
            supported_datasets,
            train_funcs,
            test_funcs,
            hyperparam_manager,
            other_arg_path,
            pretrain_path,
            sample_tag,
            pin_memory,
            min_memory,
            test_batch_size,
            DefaultFullyRankingTrainPipeline,
            DefaultFullyRankingTestPipeline)


class DefaultLeaveOneOutPipeline(_DefaultPipeline):
    def __init__(
            self,
            description: str,
            supported_datasets: List[str],
            train_funcs: DatasetFuncs,
            test_funcs: DatasetFuncs,
            hyperparam_manager: HyperparamManager,
            other_arg_path: Union[str, Path],
            pretrain_path: Union[str, Path],
            sample_tag: str,
            pin_memory: bool,
            min_memory: float,
            test_batch_size: int = 4096):
        super().__init__(
            description,
            supported_datasets,
            train_funcs,
            test_funcs,
            hyperparam_manager,
            other_arg_path,
            pretrain_path,
            sample_tag,
            pin_memory,
            min_memory,
            test_batch_size,
            DefaultLeaveOneOutTrainPipeline,
            DefaultLeaveOneOutTestPipeline)
