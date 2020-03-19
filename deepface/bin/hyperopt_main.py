import hyperopt
from hyperopt import hp, fmin, tpe
from hyperopt.mongoexp import MongoTrials

import hyperopt_optimizer


if __name__ == '__main__':
    exp_key = 'deepface15'
    print('---- %s ----' % exp_key)
    space = hp.choice('parameters', [
        {
            'crop_y_ratio': hp.uniform('crop_y_ratio', 0.3, 0.7),
            'size_ratio': hp.uniform('size_ratio', 1.0, 3.0),
        }
    ])
    trials = MongoTrials('mongo://hyper-mongo.devel.kakao.com:10247/curtis_db/jobs', exp_key=exp_key)
    best = fmin(hyperopt_optimizer.objective, space, trials=trials, algo=tpe.suggest, max_evals=100, verbose=1)
    print(trials.best_trial['result'])
    print(hyperopt.space_eval(space, best))
