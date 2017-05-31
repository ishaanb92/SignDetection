opencv_traincascade -data classifier -vec train.vec -bg bg.txt\
 -numStages 20 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos 800\
 -numNeg 5000 -w 80 -h 40 -mode ALL -precalcValBufSize 1024\
 -precalcIdxBufSize 1024
