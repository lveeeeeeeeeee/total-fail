# funny telegram bot
## neural networks minor 2024

### task: people segmentation
dataset: https://www.kaggle.com/datasets/nikhilroxtomar/person-segmentation
model : deeplabv3+ with resnet50 as encoder; squeeze and excite blocks to pay attention to detail

## result: FAIL
### why the fuck up?
i think the fuck up happened because of these things:
- lr = 3*10^-4 and barely decreaces
- bad loss function even though i have dice loss which i didnt use to compile the model
- train got mixed up with validation due to tensorflow crashing during training
- insane augmentation
- no actual testing done
- poor metrics chosen to monitor the model's learning (precision and recall, loss never dropped but i proactively ignored this)
- (maybe batch size?)
- (maybe dropouts everywhere with rate 0.2 and batch normalising layers?)

idk about epochs amount though (25). really no idea if it would turn out better

## telegram bot, what about it?
nothing. very simple, almost "echo" type of bot. aiogram (no webhooks).

## dependencies
mainly in the notebook. important though, 
- keras is 3.3.0 (train on 2.15.0)
- tensorflow is 2.16.0 (train on 2.15.0-cpp311-cpp311-29-29-linux-x86-64.whl or something)
- iaa thing for augmentation is from https://github.com/marcown/imgaug.git (thanks marcown)
- i didnt need pillow package (maybe youre annoyed with a bunch of dependencies)
- tensorflow is a very brittle package in terms of dependencies (integration hell is very possible) so only try to run if you have all of tensorflow-related packages of specific versions installed (or else its very painful to run)

## conclusions and results
this wont be "maintained" obviously, its a one-off
to run, just cd into the repo directory and python (and install dependencies)
results are very bad

you can see segmentation results for yourself in "results" directory. green (or cyan) colored areas are supposed to represent background

# main takeout: never use tensorflow