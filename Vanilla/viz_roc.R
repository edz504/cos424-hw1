library(ggplot2)

roc.df <- read.csv('roc_vanilla.csv')
ggplot(roc.df, aes(x=fpr, y=tpr)) + geom_line() + geom_point(size=2)
ggsave(file='roc_vanilla.png', width=10, height=6)

ggplot(roc.df, aes(x=fpr, y=tpr)) + geom_line() + 
    xlim(c(0, 0.01))
ggsave(file='roc_vanilla_zoom.png', width=10, height=6)