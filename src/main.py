"""Entry point for Bob's construction case study"""
import matplotlib.pyplot as plt
from utils import utils


def advice_1():
    score_features = ['GecombineerdeAfstandTotFaciliteitenGenormaliseerdInverted',
                      'BevolkingsdichtheidGenormaliseerdInverted',
                      'BouwgrondOppGenormaliseerd']
    score_weights = [1, 1, 1]
    data_1 = utils.preprocess_data_1(utils.load_csv('../data/data_1.csv'))
    data_1 = utils.append_combined_features(data_1)
    for gwb in ['Gemeente', 'Wijk', 'Buurt']:
        cross = data_1.xs(gwb, level='SoortRegio').copy()
        cross = utils.make_score(cross, score_features, score_weights)
        cross = utils.normalize_columns(cross, ['Score'])
        nlargest = cross[['WijkenEnBuurten', 'ScoreGenormaliseerd']].nlargest(5, columns='ScoreGenormaliseerd')
        print(gwb, nlargest, '\n')
        _, ax = plt.subplots()
        xticks = [f'{aanduiding}\n({gemeente})' for gemeente, aanduiding in zip(nlargest.index.values, nlargest['WijkenEnBuurten'].values)]
        nlargest.plot(kind='bar', y='ScoreGenormaliseerd', ax=ax)
        plt.ylabel('Percentiel')
        plt.xlabel('')
        plt.xticks(range(0, len(xticks)), xticks)
        plt.legend().remove()
        plt.suptitle(gwb)
        plt.tight_layout()
        plt.savefig(f'../output/{gwb}')
        plt.show()


def main():
    advice_1()


if __name__ == '__main__':
    main()
