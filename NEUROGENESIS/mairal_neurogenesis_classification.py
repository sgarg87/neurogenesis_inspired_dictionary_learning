import polyssifier_wrapper as pp
import json
import matplotlib.pyplot as plt
import math


font_size = 20


def process_method(dir_path, method_name, fmt_str, is_classify_update):
    mname = dir_path + method_name + '.mat'
    out_dir = dir_path
    #
    if is_classify_update:
        pp.process_main(mname, out_dir, fmt_str)
    #
    with open(dir_path + method_name + '_plots_data.json', 'r') as f_json:
        plots_map = json.load(f_json)
    #
    return plots_map


def plot_classification_results(mairal_map, neurogenesis_map, classifier_name, num_top_var_mairal, num_top_var_neurogenesis, dir_path):
    # print 'num_top_var_mairal', num_top_var_mairal
    # print 'num_top_var_neurogenesis', num_top_var_neurogenesis
    # print 'mairal_map', mairal_map
    # print 'neurogenesis_map', neurogenesis_map
    #
    min_var = min(min(num_top_var_mairal), min(num_top_var_neurogenesis))
    print 'min_var', min_var
    max_var = max(max(num_top_var_mairal), max(num_top_var_neurogenesis))
    print 'max_var', max_var
    #
    plt.close()
    #
    #
    plt.xlabel('Number of top variables', fontsize=font_size)
    plt.ylabel('Error rate', fontsize=font_size)
    plt.xscale('log', basex=2)
    plt.title(classifier_name+' Classifier', fontsize=font_size)
    plt.xlim((2**(math.log(min_var, 2)-1)), (2**(math.log(max_var, 2)+1)))
    plt.errorbar(num_top_var_mairal, mairal_map['total_error']['means'], mairal_map['total_error']['stds'], fmt='-md', label='ODL', lw=2, ms=12, mew=2, markerfacecolor='none', markeredgecolor='m')
    plt.errorbar(num_top_var_neurogenesis, neurogenesis_map['total_error']['means'], neurogenesis_map['total_error']['stds'], fmt='-bx', label='NODL', lw=2, ms=12, markerfacecolor='none', mew=2, markeredgecolor='b')
    curr_path = dir_path+classifier_name
    print 'curr_path', curr_path
    plt.legend(loc=1, prop={'size': font_size})
    plt.tick_params(labelsize=font_size)
    plt.tight_layout()
    plt.savefig(curr_path+'.png', dpi=300, format='png')

if __name__ == '__main__':
    import sys
    dir_path = sys.argv[1]
    is_classify_update = bool(sys.argv[2])
    #
    # Mairal fixed size
    mairal_plots_map = \
        process_method(
            dir_path=dir_path,
            method_name='mairal_sparse_codings_test',
            fmt_str='-x',
            is_classify_update=is_classify_update
        )
    # print 'mairal_plots_map', mairal_plots_map
    # Neurogenesis and group sparsity in Mairal
    neurogenesis_plots_map = \
        process_method(
            dir_path=dir_path,
            method_name='neurogen_group_mairal_sparse_codings_test',
            fmt_str='--s',
            is_classify_update=is_classify_update
        )
    # print 'neurogenesis_plots_map', neurogenesis_plots_map
    #
    #  generate new plots here
    num_top_var_mairal = mairal_plots_map['numTopVars']
    mairal_plots_map.pop('numTopVars', None)
    #
    num_top_var_neurogenesis = neurogenesis_plots_map['numTopVars']
    neurogenesis_plots_map.pop('numTopVars', None)
    #
    assert len(mairal_plots_map.keys()) == len(neurogenesis_plots_map.keys())
    #
    for curr_classifier_name in mairal_plots_map:
        print '**********{}*************'.format(curr_classifier_name)
        assert curr_classifier_name in neurogenesis_plots_map
        curr_mairal_map = mairal_plots_map[curr_classifier_name]
        print 'curr_mairal_map', curr_mairal_map
        curr_neurogenesis_map = neurogenesis_plots_map[curr_classifier_name]
        print 'curr_neurogenesis_map', curr_neurogenesis_map
        #
        plot_classification_results(
            mairal_map=curr_mairal_map,
            neurogenesis_map=curr_neurogenesis_map,
            classifier_name=curr_classifier_name,
            num_top_var_mairal=num_top_var_mairal,
            num_top_var_neurogenesis=num_top_var_neurogenesis,
            dir_path=dir_path
        )
