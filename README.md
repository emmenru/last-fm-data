# last-fm-data
 Digging into scrobbled music data to find recommendations

 Backup is set to Google drive since Colab is used. 

 Note that all timestamps are in UTC. 

 Tracks do not have full album coverage, seems to be at 35% currently. 
 Some tracks do not have detailed meta data (resulting e.g. in duration of 0). 

 Additional API requests are sent to find meta data for artists tags. This means that if an artist has 10 tags 10 calls will be sent. 