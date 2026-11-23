data_dict = {
    0: 'Scene-15',
    1: 'handwritten',
    2: 'NoisyMNIST',
    3:'synthetic3d',


}


def get_config(flag=1):
    """Determine the parameter information of the network"""
    data_name = data_dict[flag]
    if data_name in ['Scene-15']:
        return dict(
            dataset=data_name,
            topk=10,
            missing_rate=0.0,
            n_clusters=15,
            data_seed=0,
            training=dict(
                lr=1.0e-4,
                epoch=50,
                batch_size=128,
            ),
            Autoencoder=dict(
                gcnEncoder1=[20, 1024, 1024, 1024, 1024 // 2],
                gcnEncoder2=[59, 1024, 1024, 1024, 1024 // 2],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            )

        )

    elif data_name in ['handwritten']:
        return dict(
            dataset=data_name,
            topk=10,
            missing_rate=0.0,
            n_clusters=10,
            data_seed=5,
            training=dict(
                lr=1.0e-3,
                epoch=100,
                batch_size=256,
            ),
            Autoencoder=dict(
                gcnEncoder1=[76, 1024, 1024, 1024, 1024 // 2],
                gcnEncoder2=[64, 1024, 1024, 1024, 1024 // 2],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
        )
    elif data_name in ['NoisyMNIST']:
        return dict(
            dataset=data_name,
            topk=10,
            missing_rate=0.0,
            n_clusters=10,
            data_seed=1,
            training=dict(
                lr=1.0e-4,
                epoch=350,
                batch_size=256,
            ),
            Autoencoder=dict(
                gcnEncoder1=[784, 1024, 1024, 1024, 1024 // 8],
                gcnEncoder2=[784, 1024, 1024, 1024, 1024 // 8],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
        )

    elif data_name in ['synthetic3d']:
        return dict(
            dataset=data_name,
            topk=10,
            missing_rate=0.0,
            n_clusters=3,
            data_seed=9,
            training=dict(
                lr=1.0e-3,
                epoch=100,
                batch_size=256,
                N=600,
            ),
            Autoencoder=dict(
                gcnEncoder1=[3, 1024, 1024, 1024, 1024 // 8],
                gcnEncoder2=[3, 1024, 1024, 1024, 1024 // 8],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
           ),
        )










