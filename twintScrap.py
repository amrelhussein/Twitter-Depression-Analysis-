import twint

# Configure
c = twint.Config()
c.Search = "depressed"
c.Limit = 1000
c.Format = "tweet id:{id}| Date: {date} |Time: {time}|Tweet: {tweet}"
c.Store_csv = True
c.Output = "depressed.csv"
c.Lang = 'en'
# Run
twint.run.Search(c)
