/*
timing.h


 */

#define TIMING_H

#include <sys/time.h>


float elapsedTime(struct timeval now, struct timeval then);

double currentTime();
