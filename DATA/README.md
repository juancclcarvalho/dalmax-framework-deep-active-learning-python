### Dataset
This project uses the [CIFAR 10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. 
Download the dataset zip CIFAR10 from [https://drive.google.com/file/d/1xNQS9QngkoxOPyQhtG6PzbAr9b83Bmk7/view?usp=sharing](https://drive.google.com/file/d/1xNQS9QngkoxOPyQhtG6PzbAr9b83Bmk7/view?usp=sharing). It must be unzipped inside the `DATA` folder.

#### Steps dataset
 - Create folder DATA in root of project.
    ```bash
        mkdir DATA
    ```
 - Download the dataset zip CIFAR10 from [https://drive.google.com/file/d/1xNQS9QngkoxOPyQhtG6PzbAr9b83Bmk7/view?usp=sharing](https://drive.google.com/file/d/1xNQS9QngkoxOPyQhtG6PzbAr9b83Bmk7/view?usp=sharing).
 - Unzip the dataset inside the `DATA` folder.
    ```bash
        unzip DATA_CIFAR10.zip -d DATA
    ```

Structure of the dataset:
```
DATA/
    DATA_CIFAR10/
        train/
            0/
                1000.png
                1001.png
                ...
            1/
            2/
            ...
            8/
            9/
        test/
            0/
                1000.png
                1001.png
                ...
            1/
            2/
            ...
            8/
            9/
```