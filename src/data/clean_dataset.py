def clean(dataset):
    return dataset.remove_columns(
                [
                    'id',
                    'topic',
                    'domain',
                    'category',
                    'difficulty'
                ]
            )
