library(ggplot2)

roc.df <- read.csv('roc_custom.csv')

ggplot(roc.df, aes(x=fpr, y=tpr, colour=model)) + geom_line()
ggsave(file='roc_custom.png', width=10, height=6)
