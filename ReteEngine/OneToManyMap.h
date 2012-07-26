#pragma once

#include <vector>
#include <map>

template<class K,class V>
class OneToManyMap
{
public:
	typedef typename std::vector<V> listvalue;
	OneToManyMap(void){}
	virtual ~OneToManyMap(void){}

	virtual bool empty(){return m_map.empty();}

	virtual void clear()
	{
		for (typename std::map<K,listvalue>::iterator it = m_map.begin(); it != m_map.end(); it++)
		{
			listvalue& curList = it->second;
			curList.clear();
		}
		m_map.clear();
	}

	virtual void put(K key, V value)
	{
		if (m_map.find(key) == m_map.end())
		{
			m_map[key].push_back(value);
		}
		else
		{
			listvalue& curList = m_map[key];

			typename listvalue::iterator it = curList.begin();
			for (; it != curList.end(); it++)
			{
				if (*it == value)
				{
					break;
				}
			}
			if (it == curList.end())
			{
				curList.push_back(value);
			}
		}
	}

	virtual listvalue* get(K key)
	{
		if (m_map.find(key) == m_map.end())
		{
			return NULL;
		}
		else
		{
			return &m_map[key];
		}		
	}

protected:
	std::map<K, listvalue> m_map;
};
